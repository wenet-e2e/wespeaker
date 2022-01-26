#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e

# The trials file is downloaded by local/make_voxceleb1.pl.
voxceleb1_trials_O=/dockerdata/hongjiwang/sid/my_sv/data/vox1/trials/vox1_O_cleaned.kaldi
voxceleb1_trials_E=/dockerdata/hongjiwang/sid/my_sv/data/vox1/trials/vox1_E_cleaned.kaldi
voxceleb1_trials_H=/dockerdata/hongjiwang/sid/my_sv/data/vox1/trials/vox1_H_cleaned.kaldi

exp=nothing
stage=0
train_name=vox2_dev
eval_name=vox1

. utils/parse_options.sh

emb_dir=$exp/embeddings

train_emb_dir=$emb_dir/$train_name
eval_emb_dir=$emb_dir/$eval_name
mean_dir=$train_emb_dir

[ ! -d $exp/scores.kaldi ] && mkdir -p $exp/scores.kaldi

if [ $stage -le 1 ] && [ ! -f ${mean_dir}/mean.vec ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $train_emb_dir/log/compute_mean.log \
    ivector-mean scp:$train_emb_dir/xvector.scp \
    $train_emb_dir/mean.vec || exit 1;
  echo "Computed mean.vec of train set."

  $train_cmd $eval_emb_dir/log/compute_mean.log \
    ivector-mean scp:$eval_emb_dir/xvector.scp \
    $eval_emb_dir/mean.vec || exit 1;
  echo "Computed mean.vec of test set."
fi

if [ $stage -le 2 ]; then
  $train_cmd ${exp}/scores.kaldi/log/vox1_testO_cos_scoring.log \
    ivector-compute-dot-products  "cat '$voxceleb1_trials_O' | cut -d\  --fields=1,2 |" \
    "ark:ivector-subtract-global-mean ${mean_dir}/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${mean_dir}/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    ${exp}/scores.kaldi/scores_cos_vox1_testO || exit 1;
    #"ark:copy-vector scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    #"ark:copy-vector scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    #${exp}/scores.kaldi/scores_cos_vox1_testO || exit 1;

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials_O ${exp}/scores.kaldi/scores_cos_vox1_testO) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 ${exp}/scores.kaldi/scores_cos_vox1_testO $voxceleb1_trials_O 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 ${exp}/scores.kaldi/scores_cos_vox1_testO $voxceleb1_trials_O 2> /dev/null`
  echo "TEST O: EER: $eer%" | tee ${exp}/scores.kaldi/vox1_cos_result
  echo "minDCF(p-target=0.01): $mindcf1" | tee -a ${exp}/scores.kaldi/vox1_cos_result
  echo "minDCF(p-target=0.001): $mindcf2" | tee -a ${exp}/scores.kaldi/vox1_cos_result
fi

if [ $stage -le 3 ]; then
  $train_cmd ${exp}/scores.kaldi/log/voxceleb1_testE_scoring.log \
    ivector-compute-dot-products  "cat '$voxceleb1_trials_E' | cut -d\  --fields=1,2 |" \
    "ark:ivector-subtract-global-mean ${mean_dir}/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${mean_dir}/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    ${exp}/scores.kaldi/scores_cos_vox1_testE || exit 1;

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials_E ${exp}/scores.kaldi/scores_cos_vox1_testE) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 ${exp}/scores.kaldi/scores_cos_vox1_testE $voxceleb1_trials_E 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 ${exp}/scores.kaldi/scores_cos_vox1_testE $voxceleb1_trials_E 2> /dev/null`
  echo "TEST E: EER: $eer%" | tee -a ${exp}/scores.kaldi/vox1_cos_result
  echo "minDCF(p-target=0.01): $mindcf1" | tee -a ${exp}/scores.kaldi/vox1_cos_result
  echo "minDCF(p-target=0.001): $mindcf2" | tee -a ${exp}/scores.kaldi/vox1_cos_result
fi

if [ $stage -le 4 ]; then
  $train_cmd ${exp}/scores.kaldi/log/voxceleb1_testH_scoring.log \
    ivector-compute-dot-products  "cat '$voxceleb1_trials_H' | cut -d\  --fields=1,2 |" \
    "ark:ivector-subtract-global-mean ${mean_dir}/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${mean_dir}/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    ${exp}/scores.kaldi/scores_cos_vox1_testH || exit 1;

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials_H ${exp}/scores.kaldi/scores_cos_vox1_testH) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 ${exp}/scores.kaldi/scores_cos_vox1_testH $voxceleb1_trials_H 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 ${exp}/scores.kaldi/scores_cos_vox1_testH $voxceleb1_trials_H 2> /dev/null`
  echo "TEST H: EER: $eer%" | tee -a ${exp}/scores.kaldi/vox1_cos_result
  echo "minDCF(p-target=0.01): $mindcf1" | tee -a ${exp}/scores.kaldi/vox1_cos_result
  echo "minDCF(p-target=0.001): $mindcf2" | tee -a ${exp}/scores.kaldi/vox1_cos_result
fi

