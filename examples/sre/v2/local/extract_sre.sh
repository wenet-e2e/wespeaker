#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

exp_dir=''
model_path=''
nj=4
gpus="[0,1]"
data_type="shard"  # shard/raw/feat
data=data
reverb_data=data/rirs/lmdb
noise_data=data/musan/lmdb
aug_plda_data=0

. tools/parse_options.sh
set -e

if [ $aug_plda_data = 0 ];then
    sre_plda_data=sre
else
    sre_plda_data=sre_aug
fi

data_name_array=(
    "${sre_plda_data}"
    "sre16_major"
    "sre16_eval_enroll"
    "sre16_eval_test"
)
data_list_path_array=(
    "${data}/${sre_plda_data}/${data_type}.list"
    "${data}/sre16_major/${data_type}.list"
    "${data}/sre16_eval_enroll/${data_type}.list"
    "${data}/sre16_eval_test/${data_type}.list"
)
data_scp_path_array=(
    "${data}/${sre_plda_data}/wav.scp"
    "${data}/sre16_major/wav.scp"
    "${data}/sre16_eval_enroll/wav.scp"
    "${data}/sre16_eval_test/wav.scp"
) # to count the number of wavs
nj_array=($nj $nj $nj $nj)
batch_size_array=(1 1 1 1) # batch_size of test set must be 1 !!!
num_workers_array=(1 1 1 1)
if [ $aug_plda_data = 0 ];then
    aug_prob_array=(0.0 0.0 0.0 0.0)
else
    aug_prob_array=(0.67 0.0 0.0 0.0)
fi
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
    --aug_prob ${aug_prob_array[$i]} \
    --reverb_data ${reverb_data} \
    --noise_data ${noise_data} \
    --nj ${nj_array[$i]} \
    --gpus $gpus
done

wait

echo "mean vector of enroll"
python tools/vector_mean.py \
  --spk2utt ${data}/sre16_eval_enroll/spk2utt \
  --xvector_scp $exp_dir/embeddings/sre16_eval_enroll/xvector.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre16_eval_enroll/enroll_spk_xvector.ark

mkdir -p ${exp_dir}/embeddings/eval
cat ${exp_dir}/embeddings/sre16_eval_enroll/enroll_spk_xvector.scp \
    ${exp_dir}/embeddings/sre16_eval_test/xvector.scp \
    > ${exp_dir}/embeddings/eval/xvector.scp

echo "Embedding dir is (${exp_dir}/embeddings)."
