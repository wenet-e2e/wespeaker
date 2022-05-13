#!/bin/bash
# coding:utf-8
# Author: Hongji Wang

. ./path.sh

stage=-1
stop_stage=-1

config=conf/ecapa_tdnn.yaml
exp_dir=exp/ECAPA_TDNN_GLOB_c512-ASTP-emb192-fbank80-num_frms200-vox2_dev-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150
gpus="[0,1]"
num_avg=10
checkpoint=
score_norm_method="asnorm"  # asnorm/snorm
top_n=100

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Preparing datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 4
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  echo "Extract embeddings ..."

  local/extract_vox.sh --exp_dir $exp_dir --model_path $avg_model --nj 4 --gpus $gpus
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
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

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
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

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "score norm ..."
  local/score_norm.sh \
    --stage 0 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --exp_dir $exp_dir
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi
