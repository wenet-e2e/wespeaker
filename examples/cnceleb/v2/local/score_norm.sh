#!/bin/bash
# coding:utf-8
# Author: Chengdong Liang

score_norm_method="asnorm"  # asnorm/snorm
cohort_set=cnceleb_train
top_n=100
exp_dir=
trials="CNC-Eval-Core.lst"

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "compute mean xvector"
  python tools/vector_mean.py \
    --spk2utt data/${cohort_set}/spk2utt \
    --xvector_scp $exp_dir/embeddings/${cohort_set}/xvector.scp \
    --spk_xvector_ark $exp_dir/embeddings/${cohort_set}/spk_xvector.ark
fi

output_name=${cohort_set}_${score_norm_method}
[ "${score_norm_method}" == "asnorm" ] && output_name=${output_name}${top_n}
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "compute norm score"
  for x in $trials; do
    python wespeaker/bin/score_norm.py \
      --score_norm_method $score_norm_method \
      --top_n $top_n \
      --trial_score_file $exp_dir/scores/${x}.score \
      --score_norm_file $exp_dir/scores/${output_name}_${x}.score \
      --cohort_emb_scp ${exp_dir}/embeddings/${cohort_set}/spk_xvector.scp \
      --eval_emb_scp ${exp_dir}/embeddings/eval/xvector.scp \
      --mean_vec_path ${exp_dir}/embeddings/cnceleb_train/mean_vec.npy
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
      2>&1 | tee -a ${scores_dir}/cnc_${score_norm_method}${top_n}_result

    python wespeaker/bin/compute_det.py \
      ${scores_dir}/${output_name}_${x}.score
  done
fi
