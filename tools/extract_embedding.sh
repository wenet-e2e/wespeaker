#!/bin/bash

# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
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

exp_dir='exp/XVEC'
model_path='avg_model.pt'
data_type='shard'  # shard/raw/feat
data_list='shard.list'  # shard.list/raw.list/feat.list
wavs_num=
store_dir=
batch_size=1
num_workers=1
nj=4
reverb_data=data/rirs/lmdb
noise_data=data/musan/lmdb
aug_prob=0.0
gpus="[0,1]"

. tools/parse_options.sh
set -e

embed_dir=${exp_dir}/embeddings/${store_dir}
log_dir=${embed_dir}/log
[ ! -d ${log_dir} ] && mkdir -p ${log_dir}

# split the data_list file into sub_file, then we can use multi-gpus to extract embeddings
data_num=$(wc -l ${data_list} | awk '{print $1}')
subfile_num=$(($data_num / $nj + 1))
split -l ${subfile_num} -d -a 3 ${data_list} ${log_dir}/split_
num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
gpus=(`echo $gpus | cut -d '[' -f2 | cut -d ']' -f1 | tr ',' ' '`)

for suffix in $(seq 0 $(($nj - 1))); do
  idx=$[$suffix % $num_gpus]
  suffix=$(printf '%03d' $suffix)
  data_list_subfile=${log_dir}/split_${suffix}
  embed_ark=${embed_dir}/xvector_${suffix}.ark
  CUDA_VISIBLE_DEVICES=${gpus[$idx]} python -u wespeaker/bin/extract.py \
    --config ${exp_dir}/config.yaml \
    --model_path ${model_path} \
    --data_type ${data_type} \
    --data_list ${data_list_subfile} \
    --embed_ark ${embed_ark} \
    --batch-size ${batch_size} \
    --num-workers ${num_workers} \
    --reverb_data ${reverb_data} \
    --noise_data ${noise_data} \
    --aug-prob ${aug_prob} \
    >${log_dir}/split_${suffix}.log 2>&1 &
done

wait

cat ${embed_dir}/xvector_*.scp >${embed_dir}/xvector.scp
embed_num=$(wc -l ${embed_dir}/xvector.scp | awk '{print $1}')
if [ $embed_num -eq $wavs_num ]; then
  echo "Successfully extract embedding for ${store_dir}" | tee ${embed_dir}/extract.result
else
  echo "Failed to extract embedding for ${store_dir}" | tee ${embed_dir}/extract.result
fi
