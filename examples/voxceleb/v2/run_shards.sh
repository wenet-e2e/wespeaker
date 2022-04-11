#!/bin/bash
# coding:utf-8
# Author: Hongji Wang

. ./path.sh

stage=-1
stop_stage=-1

# config=conf/ecapa_tdnn.yaml
# exp_dir=exp/ECAPA_TDNN_GLOB_c512-ASTP-emb192-fbank80-num_frms200-vox2_dev-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150
# config=conf/resnet.yaml
config=conf/resnet_uio.yaml
exp_dir=exp/ResNet34-TSTP-emb256-fbank80-num_frms200-vox2_dev-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_UIO_0410

gpus="[2,3]"
num_avg=10
# checkpoint=exp/ResNet34-TSTP-emb256-fbank80-num_frms200-vox2_dev-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_UIO_0404/models/model_25.pt
checkpoint=

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Preparing datasets ..."
  ./local/prepare_data.sh --stage 4 --stop_stage 4
  cat data/vox2_dev/utt2spk | awk '{print $2}' | sort | uniq | \
      awk '{print $1, NR - 1}' > data/vox2_dev/spk2id
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert training data to shard ..."
  python tools/make_shard_list.py --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --seed '777' \
      --shuffle \
      data/vox2_dev/wav.scp data/vox2_dev/utt2spk \
      data/vox2_dev/shards data/vox2_dev/shard.list
  # Convert all musan data to LMDB
  python tools/make_lmdb.py data/musan/wav.scp data/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py data/rirs/wav.scp data/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train_uio.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --train_list data/vox2_dev/shard.list \
      --spk2id data/vox2_dev/spk2id \
      --reverb_lmdb data/rirs/lmdb \
      --noise_lmdb data/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  echo "Extract embeddings ..."

  local/extract_vox.sh --exp_dir $exp_dir --model_path $avg_model --nj 4 --gpus $gpus
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Apply cosine scoring ..."
  mkdir -p ${exp_dir}/scores
  trials_dir=data/vox1/trials
  python wespeaker/bin/score.py \
    --exp_dir ${exp_dir} \
    --eval_scp_path ${exp_dir}/embeddings/vox1/xvector.scp \
    --cal_mean True \
    --cal_mean_dir ${exp_dir}/embeddings/vox2_dev \
    ${trials_dir}/vox1_O_cleaned.kaldi \
    ${trials_dir}/vox1_E_cleaned.kaldi \
    ${trials_dir}/vox1_H_cleaned.kaldi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Compute metrics (EER/minDCF) ..."
  scores_dir=${exp_dir}/scores
  python wespeaker/bin/compute_metrics.py \
    --p_target 0.01 \
    --c_fa 1 \
    --c_miss 1 \
    ${scores_dir}/vox1_O_cleaned.kaldi.score \
    ${scores_dir}/vox1_E_cleaned.kaldi.score \
    ${scores_dir}/vox1_H_cleaned.kaldi.score \
    2>&1 | tee ${scores_dir}/vox1_cos_result

  echo "Compute DET curve ..."
  python wespeaker/bin/compute_det.py \
    ${scores_dir}/vox1_O_cleaned.kaldi.score \
    ${scores_dir}/vox1_E_cleaned.kaldi.score \
    ${scores_dir}/vox1_H_cleaned.kaldi.score
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi
