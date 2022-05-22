#!/bin/bash
# coding:utf-8
# Author: Hongji Wang

exp_dir=''
model_path=''
nj=4
gpus="[0,1]"
data_type="shard/raw"  # shard/raw

. tools/parse_options.sh
set -e

# wav.scp is to count the number of utts
data_scp_path_array=("data/vox2_dev/wav.scp" "data/vox1/wav.scp")
data_list_path_array=("data/vox2_dev/${data_type}.list" "data/vox1/${data_type}.list")
utt2spk_path_array=("data/vox2_dev/utt2spk" "data/vox1/utt2spk")
#data_path_array=("data/vox2_dev/feats.scp" "data/vox1/feats.scp")
#raw_wav_array=(False False)

data_name_array=("vox2_dev" "vox1")
nj_array=($nj $nj)
batch_size_array=(16 1) # batch_size of test set must be 1 !!!
num_workers_array=(4 1)
count=${#data_name_array[@]}

for i in $(seq 0 $(($count - 1))); do
  bash tools/extract_embedding.sh --exp_dir ${exp_dir} \
    --model_path $model_path \
    --data_scp ${data_scp_path_array[$i]} \
    --data_type ${data_type} \
    --data_list ${data_list_path_array[$i]} \
    --utt2spk ${utt2spk_path_array[$i]} \
    --store_dir ${data_name_array[$i]} \
    --batch_size ${batch_size_array[$i]} \
    --num_workers ${num_workers_array[$i]} \
    --nj ${nj_array[$i]} \
    --gpus $gpus &
done

wait

echo "Embedding dir is (${exp_dir}/embeddings)."
