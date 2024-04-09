#!/bin/bash

# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2023 Zhengyang Chen (chenhzhengyang117@gmail.com)
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

#exp_dir=
#trials="trials trials_tgl trials_yue"
#data=data


trials=""
xvectors=""
cal_mean_dir=""
exp_dir=""

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

echo "  - trials $trials"
echo "  - xvectors $xvectors"
echo "  - cal_mean dir $cal_mean_dir"
echo "  - exp_dir $exp_dir"

output_name=$(echo $cal_mean_dir | sed "s:.*embeddings/::" | sed -e "s:/:_:g")
scores_dir=${exp_dir}/scores

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "apply cosine scoring ..."
  mkdir -p ${exp_dir}/scores
  for x in $(echo $trials | tr "," " "); do
    echo $x
    python wespeaker/bin/score.py \
      --exp_dir ${exp_dir} \
      --eval_scp_path $xvectors \
      --cal_mean True \
      --cal_mean_dir $cal_mean_dir \
      ${x}
    xx=$(basename  $x)  
    mv ${scores_dir}/${xx}.score ${scores_dir}/${xx}.mean_${output_name}_cos.score
  done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "compute metrics (EER/minDCF) ..."
  for x in $(echo $trials | tr "," " "); do
    xx=$(basename  $x)  
    echo $xx
    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir}/${xx}.mean_${output_name}_cos.score \
        2>&1 | tee ${scores_dir}/${xx}.mean_${output_name}_cos.result

    echo "compute DET curve ..."
    python wespeaker/bin/compute_det.py \
        ${scores_dir}/${xx}.mean_${output_name}_cos.score
  done
fi
