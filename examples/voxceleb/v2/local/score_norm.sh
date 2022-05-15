#!/bin/bash
# coding:utf-8
# Author: Chengdong Liang

score_norm_method="snorm"  # asnorm/snorm
cohort_set=vox2_dev
top_n=100
exp_dir=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "split enroll and test from trials"
  # voxceleb -> voxceleb1-O/E/H[-clean]_enroll/test
  trials_dir=data/vox1/trials
  for x in ${trials}; do
    awk '{print $1}' $trials_dir/$x | sort -u > data/vox1/${x}_enroll.list
    awk '{print $2}' $trials_dir/$x | sort -u > data/vox1/${x}_test.list
  done

  echo "get corhot.list"
  awk '{print $1}' data/$cohort_set/spk2utt >data/$cohort_set/cohort.list
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "compute mean xvector"
  python tools/vector_mean.py \
    --spk2utt data/$cohort_set/spk2utt \
    --xvector_scp $exp_dir/embeddings/vox2_dev/xvector.scp \
    --spk_xvector_ark $exp_dir/embeddings/vox2_dev/spk_xvector.ark
fi

output_name=${cohort_set}_${score_norm_method}
[ "${score_norm_method}" == "asnorm" ] && output_name=${output_name}${top_n}
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "compute norm score"
  for x in $trials; do
    python wespeaker/utils/score_norm.py \
      --enroll_list_file data/vox1/${x}_enroll.list \
      --test_list_file data/vox1/${x}_test.list \
      --cohort_list_file data/$cohort_set/cohort.list \
      --score_norm_method $score_norm_method \
      --top_n $top_n \
      --trials_score_file $exp_dir/scores/${x}.score \
      --score_norm_file $exp_dir/scores/${output_name}_${x}.score \
      --cal_mean True \
      --mean_path ${exp_dir}/embeddings/vox2_dev/mean_vec.npy \
      --cohort_emb_scp ${exp_dir}/embeddings/vox2_dev/spk_xvector.scp \
      --eval_emb_scp ${exp_dir}/embeddings/vox1/xvector.scp
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "compute metrics"
  for x in ${trials}; do
    scores_dir=${exp_dir}/scores
    python wespeaker/bin/compute_metrics.py \
      --p_target 0.01 \
      --c_fa 1 \
      --c_miss 1 \
      ${scores_dir}/${output_name}_${x}.score \
      2>&1 | tee -a ${scores_dir}/vox1_${score_norm_method}${top_n}_result_0512

    python wespeaker/bin/compute_det.py \
      ${scores_dir}/${output_name}_${x}.score
  done
fi
