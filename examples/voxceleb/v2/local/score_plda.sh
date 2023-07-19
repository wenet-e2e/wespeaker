#!/bin/bash

# Copyright (c) 2022 Shuai Wang (wsstriving@gmail.com)
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
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "train the plda model ..."
  python wespeaker/bin/train_plda.py \
    --exp_dir ${exp_dir} \
    --scp_path ${exp_dir}/embeddings/vox2_dev/xvector.scp \
    --utt2spk ${data}/vox2_dev/utt2spk \
    --indim 256 \
    --iter 5
  echo "plda training finished"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "apply plda scoring ..."
  mkdir -p ${exp_dir}/scores
  trials_dir=${data}/vox1/trials
  for x in $trials; do
    echo "scoring on " $x
    python wespeaker/bin/eval_plda.py \
      --enroll_scp_path ${exp_dir}/embeddings/vox1/xvector.scp \
      --test_scp_path ${exp_dir}/embeddings/vox1/xvector.scp \
      --utt2spk <(cat ${data}/vox1/utt2spk | awk '{print $1, $1}') \
      --trial ${trials_dir}/${x} \
      --score_path ${exp_dir}/scores/${x}.pldascore \
      --model_path ${exp_dir}/plda
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir=${exp_dir}/scores
  for x in $trials; do
    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir}/${x}.pldascore \
        2>&1 | tee -a ${scores_dir}/vox1_plda_result

    echo "compute DET curve ..."
    python wespeaker/bin/compute_det.py \
        ${scores_dir}/${x}.pldascore
  done
fi
