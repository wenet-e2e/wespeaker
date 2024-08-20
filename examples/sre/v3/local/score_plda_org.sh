#!/bin/bash

# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)
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
exp_dir=
data=data
trials="${data}/sre16/eval/trials ${data}/sre16/eval/trials_tgl ${data}/sre16/eval/trials_yue"
aug_plda_data=0

enroll_scp=sre16/eval/enrollment/xvector.scp
test_scp=sre16/eval/test/xvector.scp
indomain_scp=sre16/major/xvector.scp    # For mean subtraction
utt2spk=data/sre16/eval/enrollment/utt2spk

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

if [ $aug_plda_data = 0 ];then
    sre_plda_data=cts
else
    sre_plda_data=cts_aug
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "train the plda model ..."
  python wespeaker/bin/train_plda.py \
    --exp_dir ${exp_dir} \
    --scp_path ${exp_dir}/embeddings/${sre_plda_data}/xvector.scp \
    --utt2spk ${data}/${sre_plda_data}/utt2spk \
    --indim 256 \
    --iter 200
  echo "plda training finished"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "apply plda scoring ..."
  mkdir -p ${exp_dir}/scores
  for x in $(echo $trials | tr "," " "); do
    xx=$(basename  $x)
    echo "scoring on " $x
    python wespeaker/bin/eval_plda.py \
      --enroll_scp_path ${exp_dir}/embeddings/$enroll_scp \
      --test_scp_path ${exp_dir}/embeddings/$test_scp \
      --indomain_scp ${exp_dir}/embeddings/$indomain_scp \
      --utt2spk $utt2spk \
      --trial ${x} \
      --score_path ${exp_dir}/scores/${xx}.pldascore \
      --model_path ${exp_dir}/plda
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "compute metrics (EER/minDCF) ..."
    scores_dir=${exp_dir}/scores
    for x in $(echo $trials | tr "," " "); do
        xx=$(basename  $x)
        python wespeaker/bin/compute_metrics.py \
            --p_target 0.01 \
            --c_fa 1 \
            --c_miss 1 \
            ${scores_dir}/${xx}.pldascore \
            2>&1 | tee -a ${scores_dir}/${xx}_plda_result

        echo "compute DET curve ..."
        python wespeaker/bin/compute_det.py \
            ${scores_dir}/${xx}.pldascore
    done
fi
