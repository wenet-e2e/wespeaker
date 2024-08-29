#!/bin/bash

# Copyright (c) 2024 Johan Rohdin (rohdin@fit.vutbr.cz)
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


# The preprocessed embeddings are already stored but we need to create the lists
# as score.sh wants them.

exp_dir=$1
data=data

# We have three different preprocessors for which we need to prepare the lists
# embd_proc_cts_aug.pkl             # LDA and cts_aug mean subtraction
# embd_proc_sre16_major.pkl         # LDA and sre16_major mean subtracion (Only used for SRE16)
# embd_proc_sre18_dev_unlabeled.pkl # LDA and sre18_dev_unlabeled mean subtracion (Only used for SRE18)


### !!!
# Note that xvector2 is only a hack for BUT

##################################################################
# CTS AUG for all sets
echo "mean vector of enroll"
python tools/vector_mean.py \
  --spk2utt ${data}/sre16/eval/enrollment/spk2utt \
  --xvector_scp $exp_dir/embeddings/sre16/eval/enrollment/xvector_proc_embd_proc_cts_aug.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre16/eval/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre18/dev/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre18/dev/enrollment/xvector_proc_embd_proc_cts_aug.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre18/dev/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre18/eval/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre18/eval/enrollment/xvector_proc_embd_proc_cts_aug.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre18/eval/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre21/dev/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre21/dev/enrollment/xvector_proc_embd_proc_cts_aug.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre21/dev/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre21/eval/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre21/eval/enrollment/xvector_proc_embd_proc_cts_aug.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre21/eval/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.ark


# Create one scp with both enroll and test since this is expected by some scripts
cat ${exp_dir}/embeddings/sre16/eval/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.scp \
    ${exp_dir}/embeddings/sre16/eval/test/xvector_proc_embd_proc_cts_aug.scp \
    > ${exp_dir}/embeddings/sre16/eval/xvector_proc_embd_proc_cts_aug.scp

cat ${exp_dir}/embeddings/sre18/dev/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.scp \
    ${exp_dir}/embeddings/sre18/dev/test/xvector_proc_embd_proc_cts_aug.scp \
    > ${exp_dir}/embeddings/sre18/dev/xvector_proc_embd_proc_cts_aug.scp

cat ${exp_dir}/embeddings/sre18/eval/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.scp \
    ${exp_dir}/embeddings/sre18/eval/test/xvector_proc_embd_proc_cts_aug.scp \
    > ${exp_dir}/embeddings/sre18/eval/xvector_proc_embd_proc_cts_aug.scp

cat ${exp_dir}/embeddings/sre21/dev/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.scp \
    ${exp_dir}/embeddings/sre21/dev/test/xvector_proc_embd_proc_cts_aug.scp \
    > ${exp_dir}/embeddings/sre21/dev/xvector_proc_embd_proc_cts_aug.scp

cat ${exp_dir}/embeddings/sre21/eval/enrollment/enroll_spk_xvector_proc_embd_proc_cts_aug.scp \
    ${exp_dir}/embeddings/sre21/eval/test/xvector_proc_embd_proc_cts_aug.scp \
    > ${exp_dir}/embeddings/sre21/eval/xvector_proc_embd_proc_cts_aug.scp


##################################################################
# sre16_major for sre16 eval
echo "mean vector of enroll"
python tools/vector_mean.py \
  --spk2utt ${data}/sre16/eval/enrollment/spk2utt \
  --xvector_scp $exp_dir/embeddings/sre16/eval/enrollment/xvector_proc_embd_proc_sre16_major.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre16/eval/enrollment/enroll_spk_xvector_proc_embd_proc_sre16_major.ark

# Create one scp with both enroll and test since this is expected by some scripts
cat ${exp_dir}/embeddings/sre16/eval/enrollment/enroll_spk_xvector_proc_embd_proc_sre16_major.scp \
    ${exp_dir}/embeddings/sre16/eval/test/xvector_proc_embd_proc_sre16_major.scp \
    > ${exp_dir}/embeddings/sre16/eval/xvector_proc_embd_proc_sre16_major.scp


##################################################################
# sre18_dev_unlabeled for sre18 dev/eval
echo "mean vector of enroll"
python tools/vector_mean.py \
  --spk2utt ${data}/sre18/dev/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre18/dev/enrollment/xvector_proc_embd_proc_sre18_dev_unlabeled.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre18/dev/enrollment/enroll_spk_xvector_proc_embd_proc_sre18_dev_unlabeled.ark

python tools/vector_mean.py \
  --spk2utt ${data}/sre18/eval/enrollment/mdl_id2utt \
  --xvector_scp $exp_dir/embeddings/sre18/eval/enrollment/xvector_proc_embd_proc_sre18_dev_unlabeled.scp \
  --spk_xvector_ark $exp_dir/embeddings/sre18/eval/enrollment/enroll_spk_xvector_proc_embd_proc_sre18_dev_unlabeled.ark

# Create one scp with both enroll and test since this is expected by some scripts
cat ${exp_dir}/embeddings/sre18/dev/enrollment/enroll_spk_xvector_proc_embd_proc_sre18_dev_unlabeled.scp \
    ${exp_dir}/embeddings/sre18/dev/test/xvector_proc_embd_proc_sre18_dev_unlabeled.scp \
    > ${exp_dir}/embeddings/sre18/dev/xvector_proc_embd_proc_sre18_dev_unlabeled.scp

cat ${exp_dir}/embeddings/sre18/eval/enrollment/enroll_spk_xvector_proc_embd_proc_sre18_dev_unlabeled.scp \
    ${exp_dir}/embeddings/sre18/eval/test/xvector_proc_embd_proc_sre18_dev_unlabeled.scp \
    > ${exp_dir}/embeddings/sre18/eval/xvector_proc_embd_proc_sre18_dev_unlabeled.scp

