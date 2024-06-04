#!/bin/bash

# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2024 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2024 Bing Han (hanbing97@sjtu.edu.cn)
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

score_norm_method="asnorm"  # asnorm/snorm
cohort_set=vox2_dev
calibration_trial="vox2_cali.kaldi"
top_n=100
exp_dir=''
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

output_name=${cohort_set}_${score_norm_method}
[ "${score_norm_method}" == "asnorm" ] && output_name=${output_name}${top_n}
trials_dir=${data}/vox1/trials

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Score calibration set"
  # Compute duration
  for dset in vox2_dev vox1; do
    if [ ! -f ${data}/${dset}/utt2dur ]; then
      python tools/wav2dur.py ${data}/${dset}/wav.scp ${data}/${dset}/utt2dur > ${data}/${dset}/dur.log
    fi
  done
  # generate trial for calibration
  if [ ! -e ${trials_dir}/${calibration_trial} ]; then
    python tools/generate_calibration_trial.py --utt2dur ${data}/vox2_dev/utt2dur --trial_path ${trials_dir}/${calibration_trial}
  fi

  python wespeaker/bin/score.py \
    --exp_dir ${exp_dir} \
    --eval_scp_path ${exp_dir}/embeddings/vox2_dev/xvector.scp \
    --cal_mean True \
    --cal_mean_dir ${exp_dir}/embeddings/vox2_dev \
    ${trials_dir}/${calibration_trial}

  python wespeaker/bin/score_norm.py \
    --score_norm_method $score_norm_method \
    --top_n $top_n \
    --trial_score_file $exp_dir/scores/${calibration_trial}.score \
    --score_norm_file $exp_dir/scores/${output_name}_${calibration_trial}.score \
    --cohort_emb_scp ${exp_dir}/embeddings/${cohort_set}/spk_xvector.scp \
    --eval_emb_scp ${exp_dir}/embeddings/vox2_dev/xvector.scp \
    --mean_vec_path ${exp_dir}/embeddings/vox2_dev/mean_vec.npy
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Prepare calibration factors"
  # gather calibration factor
  mkdir -p ${exp_dir}/scores/calibration
  cat ${data}/vox1/utt2dur ${data}/vox2_dev/utt2dur > ${exp_dir}/scores/calibration/utt2dur
  for x in ${calibration_trial} $trials; do
    python wespeaker/bin/score_calibration.py "gather_calibration_factors" \
      --wav_dur_scp ${exp_dir}/scores/calibration/utt2dur \
      --max_dur 20 \
      --score_norm_file ${exp_dir}/scores/${output_name}_${x}.score \
      --calibration_factor_file ${exp_dir}/scores/calibration/${output_name}_${x}.calibration
  done
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Train calibration model"
  python wespeaker/bin/score_calibration.py "train_calibration_model" \
    --calibration_factor_file ${exp_dir}/scores/calibration/${output_name}_${calibration_trial}.calibration \
    --save_model_path ${exp_dir}/scores/calibration/calibration_model.pt
fi

cali_output_name=cali_${output_name}
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Infer calibration model"
  for x in ${trials}; do
    python wespeaker/bin/score_calibration.py "infer_calibration" \
      --calibration_factor_file ${exp_dir}/scores/calibration/${output_name}_${x}.calibration \
      --save_model_path ${exp_dir}/scores/calibration/calibration_model.pt \
      --calibration_score_file ${exp_dir}/scores/${cali_output_name}_${x}.score
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "compute metrics"
  for x in ${trials}; do
    scores_dir=${exp_dir}/scores
    python wespeaker/bin/compute_metrics.py \
      --p_target 0.01 \
      --c_fa 1 \
      --c_miss 1 \
      ${scores_dir}/${cali_output_name}_${x}.score \
      2>&1 | tee -a ${scores_dir}/vox1_cali_${score_norm_method}${top_n}_result

    python wespeaker/bin/compute_det.py \
      ${scores_dir}/${cali_output_name}_${x}.score
  done
fi
