#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)

# Adapted by: 2024 Ondrej Odehnal OndrejOdehnal42@gmail.com

. ./path.sh || exit 1

stage=3
stop_stage=3

# data=data
# data="/scratch/project/open-28-58/xodehn09/data"
# data="/mnt/proj3/open-27-67/xodehn09/data/16kHz/NAKI/SPLIT"
data="${DATA_DIR}"

base_port=29401
max_port=40000
current_time=$(date +%s)
PORT=$((current_time % (max_port - base_port) + base_port))

export HOST_NODE_ADDR=0.0.0.0:$PORT
export OMP_NUM_THREADS=32
# export LOGLEVEL=DEBUG

data_type="raw"  # shard/raw

# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40
# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40-no-margin
# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40-softmax
# exp_name=WavLM-BasePlus-Last_ASTP-emb257-3s-LRS10-Epoch20-no-margin
# exp_name=WavLM-BasePlus-LWAP_Mean-emb257-3s-LRS10-Epoch20-no-margin
# exp_name=WavLM-BasePlus-LWAP_PoolDim-emb257-3s-LRS10-Epoch20-no-margin

# WavLM pre-trained
# config=
# exp_dir=

gpus="[0]"
# checkpoint=

# WavLM pre-trained
# exp_dir=exp/$exp_name
# config="$exp_dir/config.yaml"
# eval_model=avg_model.pt

num_avg=2
# checkpoint=$exp_dir/models/model_20.pt


# setup for large margin fine-tuning
# lm_config=conf/wavlm_base_MHFA_LR_lm.yaml

. tools/parse_options.sh || exit 1

echo "GPUs: $gpus"
echo "exp_dir: $exp_dir"
echo "config: $config"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}
  model_path=$avg_model

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then


  # NAKI
  # NOTE: Create scores and prints accuracy
  echo "Evalute model ..."
  echo $exp_dir/models/$eval_model
  model_path=$exp_dir/models/$eval_model
  tools/evaluate_V2.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 1 --gpus $gpus --data_type ${data_type} \
    --data_list ${data}/NAKI_filtered/test/NAKI_split/raw.list \
    --store_dir NAKI_filtered/test \
    --data_label ${data}/NAKI_filtered/test/NAKI_split/utt2spk

  # VoxLingua107
  # # NOTE: Create scores and prints accuracy
  # echo "Evalute model ..."
  # echo $exp_dir/models/$eval_model
  # model_path=$exp_dir/models/$eval_model
  # tools/evaluate_V2.sh \
  #   --exp_dir $exp_dir --model_path $model_path \
  #   --nj 1 --gpus $gpus --data_type ${data_type} \
  #   --data_list ${data}/voxlingua107_dev/raw.list \
  #   --store_dir voxlingua107_dev \
  #   --data_label ${data}/voxlingua107_dev/utt2spk

  # echo "Extract embeddings ..."
  # local/extract_vox.sh \
  #   --exp_dir $exp_dir --model_path $model_path \
  #   --nj 4 --gpus $gpus --data_type $data_type --data ${data}


fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

  echo "Extract embeddings ..."

  model_path=$exp_dir/models/$eval_model
  local/extract_naki.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 1 --gpus $gpus --data_type ${data_type} 

  #   # --data_label ${data}/NAKI_filtered/test/NAKI_split/utt2spk


 #  model_path=$exp_dir/models/$eval_model
 #  wavs_num=$(wc -l ${data_scp_path_array[$i]} | awk '{print $1}')
 #  bash tools/extract_embedding_V2.sh --exp_dir ${exp_dir} \
 #    --model_path $model_path \
 #    --data_type ${data_type} \
 #    --data_list ${data}/NAKI_filtered/test/NAKI_split/raw.list \
 #    --wavs_num ${wavs_num} \
 #    --store_dir NAKI_filtered/test \
 #    --batch_size 1  \
 #    --num_workers 1  \
 #    --nj 4  \
 #    --gpus $gpus &

fi

