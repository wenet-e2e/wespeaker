#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2024 Johan Rohdin (rohdin@fit.vutbr.cz)
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



####
true && {
data_name_array=(
    "cts_aug"
    "sre16/major"
    "sre16/eval/enrollment"
    "sre16/eval/test"
    "sre18/dev/enrollment/"
    "sre18/dev/test/"
    "sre18/dev/unlabeled/"
    "sre18/eval/enrollment/"
    "sre18/eval/test/"
    "sre21/dev/enrollment/"
    "sre21/dev/test/"
    "sre21/eval/enrollment/"
    "sre21/eval/test/"
)
data_list_path_array=(
    "${data}/cts_aug/${data_type}.list"
    "${data}/sre16/major/${data_type}.list"
    "${data}/sre16/eval/enrollment/${data_type}.list"
    "${data}/sre16/eval/test/${data_type}.list"
    "${data}/sre18/dev/enrollment/${data_type}.list"
    "${data}/sre18/dev/test/${data_type}.list"
    "${data}/sre18/dev/unlabeled/${data_type}.list"
    "${data}/sre18/eval/enrollment/${data_type}.list"
    "${data}/sre18/eval/test/${data_type}.list"
    "${data}/sre21/dev/enrollment/${data_type}.list"
    "${data}/sre21/dev/test/${data_type}.list"
    "${data}/sre21/eval/enrollment/${data_type}.list"
    "${data}/sre21/eval/test/${data_type}.list"
)
data_scp_path_array=(
    "${data}/cts_aug/wav.scp"
    "${data}/sre16/major/wav.scp"
    "${data}/sre16/eval/enrollment/wav.scp"
    "${data}/sre16/eval/test/wav.scp"
    "${data}/sre18/dev/enrollment/wav.scp"
    "${data}/sre18/dev/test/wav.scp"
    "${data}/sre18/dev/unlabeled/wav.scp"
    "${data}/sre18/eval/enrollment/wav.scp"
    "${data}/sre18/eval/test/wav.scp"
    "${data}/sre21/dev/enrollment/wav.scp"
    "${data}/sre21/dev/test/wav.scp"
    "${data}/sre21/eval/enrollment/wav.scp"
    "${data}/sre21/eval/test/wav.scp"
) # to count the number of wavs
nj_array=($nj $nj $nj $nj $nj $nj $nj $nj $nj $nj $nj $nj $nj)
batch_size_array=(1 1 1 1 1 1 1 1 1 1 1 1 1) # batch_size of test set must be 1 !!!
num_workers_array=(1 1 1 1 1 1 1 1 1 1 1 1 1)
aug_prob_array=(0.67 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
}


count=${#data_name_array[@]}

true && {
for i in $(seq 0 $(($count - 1))); do
    echo $i
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
}

# Create enrollment models. This is the first order statistics. The zeroth order
# (the number of enrollment vectors) should, in principle, also be considered.
echo "mean vector of enroll"
python tools/vector_mean.py \
  --spk2utt ${data}/sre16/eval/enrollment/spk2utt \
  --xvector_scp $exp_dir/embeddings/sre16/eval/enrollment/xvector.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre16/eval/enrollment/enroll_spk_xvector.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre18/dev/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre18/dev/enrollment/xvector.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre18/dev/enrollment/enroll_mdl_xvector.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre18/eval/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre18/eval/enrollment/xvector.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre18/eval/enrollment/enroll_mdl_xvector.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre21/dev/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre21/dev/enrollment/xvector.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre21/dev/enrollment/enroll_mdl_xvector.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre21/eval/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre21/eval/enrollment/xvector.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre21/eval/enrollment/enroll_mdl_xvector.ark


# Create one scp with both enroll and test since this is expected by some scripts
cat ${exp_dir}/embeddings/sre16/eval/enrollment/enroll_spk_xvector.scp \
    ${exp_dir}/embeddings/sre16/eval/test/xvector.scp \
    > ${exp_dir}/embeddings/sre16/eval/xvector.scp

cat ${exp_dir}/embeddings/sre18/dev/enrollment/enroll_mdl_xvector.scp \
    ${exp_dir}/embeddings/sre18/dev/test/xvector.scp \
    > ${exp_dir}/embeddings/sre18/dev/xvector.scp

cat ${exp_dir}/embeddings/sre18/eval/enrollment/enroll_mdl_xvector.scp \
    ${exp_dir}/embeddings/sre18/eval/test/xvector.scp \
    > ${exp_dir}/embeddings/sre18/eval/xvector.scp

cat ${exp_dir}/embeddings/sre21/dev/enrollment/enroll_mdl_xvector.scp \
    ${exp_dir}/embeddings/sre21/dev/test/xvector.scp \
    > ${exp_dir}/embeddings/sre21/dev/xvector.scp

cat ${exp_dir}/embeddings/sre21/eval/enrollment/enroll_mdl_xvector.scp \
    ${exp_dir}/embeddings/sre21/eval/test/xvector.scp \
    > ${exp_dir}/embeddings/sre21/eval/xvector.scp



echo "Embedding dir is (${exp_dir}/embeddings)."


