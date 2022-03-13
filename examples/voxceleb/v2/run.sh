#!/bin/bash
# coding:utf-8
# Author: Hongji Wang

. ./path.sh

stage=-1
stop_stage=-1

config=conf/ecapa_tdnn.yaml
exp_dir=exp/ECAPA_TDNN_GLOB_c512-ASTP-emb192-fbank80-num_frms200-vox2_dev-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch66
gpus="[0,1]"
num_avg=10

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
      --num_avg ${num_avg}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  echo "Extract embeddings ..."
  # !!!IMPORTANT!!!
  # nj should not exceed num of gpus in local machine
  local/extract_vox.sh --exp_dir $exp_dir --model_path $avg_model --nj 4
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Apply cosine scoring ..."
  mkdir -p ${exp_dir}/scores
  trials_dir=data/vox1/trials
  python -u wespeaker/bin/score.py \
    --exp_dir ${exp_dir} \
    --eval_scp_path ${exp_dir}/embeddings/vox1/xvector.scp \
    --cal_mean True \
    --cal_mean_dir ${exp_dir}/embeddings/vox2_dev \
    --p_target 0.01 \
    --c_miss 1 \
    --c_fa 1 \
    ${trials_dir}/vox1_O_cleaned.kaldi ${trials_dir}/vox1_E_cleaned.kaldi ${trials_dir}/vox1_H_cleaned.kaldi \
    2>&1 | tee ${exp_dir}/scores/vox1_cos_result
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi
