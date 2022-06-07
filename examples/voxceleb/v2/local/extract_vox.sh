#!/bin/bash
# coding:utf-8
#
# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)

exp_dir=''
model_path=''
nj=4
gpus="[0,1]"
data_type="shard/raw"  # shard/raw

. tools/parse_options.sh
set -e

data_name_array=("vox2_dev" "vox1")
data_list_path_array=("data/vox2_dev/${data_type}.list" "data/vox1/${data_type}.list")
data_scp_path_array=("data/vox2_dev/wav.scp" "data/vox1/wav.scp") # to count the number of wavs
nj_array=($nj $nj)
batch_size_array=(16 1) # batch_size of test set must be 1 !!!
num_workers_array=(4 1)
count=${#data_name_array[@]}

for i in $(seq 0 $(($count - 1))); do
  wavs_num=$(wc -l ${data_scp_path_array[$i]} | awk '{print $1}')
  bash tools/extract_embedding.sh --exp_dir ${exp_dir} \
    --model_path $model_path \
    --data_type ${data_type} \
    --data_list ${data_list_path_array[$i]} \
    --wavs_num ${wavs_num} \
    --store_dir ${data_name_array[$i]} \
    --batch_size ${batch_size_array[$i]} \
    --num_workers ${num_workers_array[$i]} \
    --nj ${nj_array[$i]} \
    --gpus $gpus &
done

wait

echo "Embedding dir is (${exp_dir}/embeddings)."
