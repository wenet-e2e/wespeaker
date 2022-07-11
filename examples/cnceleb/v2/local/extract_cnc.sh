#!/bin/bash
# coding:utf-8

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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
data_type="shard/raw"  # shard/raw
data=data

. tools/parse_options.sh
set -e

data_name_array=("cnceleb_train" "eval")
data_list_path_array=("${data}/cnceleb_train/${data_type}.list" "${data}/eval/${data_type}.list")
data_scp_path_array=("${data}/cnceleb_train/wav.scp" "${data}/eval/wav.scp")
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

echo "mean vector of enroll"
python tools/vector_mean.py \
  --spk2utt ${data}/eval/enroll.map \
  --xvector_scp $exp_dir/embeddings/eval/xvector.scp \
  --spk_xvector_ark $exp_dir/embeddings/eval/enroll_spk_xvector.ark

cat $exp_dir/embeddings/eval/enroll_spk_xvector.scp >> $exp_dir/embeddings/eval/xvector.scp
