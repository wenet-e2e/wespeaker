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

# The trials file in kaldi format.
vox1_test_trials_O=/dockerdata/hongjiwang/sid/my_sv/data/vox1/trials.kaldi/vox1_O_clean.kaldi
vox1_test_trials_E=/dockerdata/hongjiwang/sid/my_sv/data/vox1/trials.kaldi/vox1_E_clean.kaldi
vox1_test_trials_H=/dockerdata/hongjiwang/sid/my_sv/data/vox1/trials.kaldi/vox1_H_clean.kaldi

exp=nothing
data=data
stage=0
train_name=vox2_dev
eval_name=vox1

. utils/parse_options.sh

embed_dir=$exp/embeddings

train_emb_dir=$embed_dir/$train_name
eval_emb_dir=$embed_dir/$eval_name
mean_dir=$train_emb_dir

mkdir -p $exp/scores.kaldi

if [ $stage -le 1 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $train_emb_dir/log/compute_mean.log \
    ivector-mean scp:$train_emb_dir/xvector.scp \
    $train_emb_dir/mean.vec || exit 1;
  echo "Computed mean.vec of train set."

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=256
  $train_cmd $train_emb_dir/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$train_emb_dir/xvector.scp ark:- |" \
    ark:${data}/${train_name}/utt2spk $train_emb_dir/transform.mat || exit 1;
  echo "Computed LDA."

  # Train the PLDA model.
  $train_cmd $train_emb_dir/log/plda.log \
    ivector-compute-plda ark:${data}/${train_name}/spk2utt \
    "ark:ivector-subtract-global-mean scp:$train_emb_dir/xvector.scp ark:- | transform-vec $train_emb_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $train_emb_dir/plda || exit 1;
  echo "Computed PLDA."
fi

if [ $stage -le 2 ]; then
  $train_cmd ${exp}/scores.kaldi/log/vox1_testO_plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $train_emb_dir/plda - |" \
    "ark:ivector-subtract-global-mean $train_emb_dir/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | transform-vec $train_emb_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $train_emb_dir/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | transform-vec $train_emb_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$vox1_test_trials_O' | cut -d\  --fields=1,2 |" ${exp}/scores.kaldi/scores_plda_vox1_testO || exit 1;

  eer=`compute-eer <(local/prepare_for_eer.py $vox1_test_trials_O ${exp}/scores.kaldi/scores_plda_vox1_testO) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 ${exp}/scores.kaldi/scores_plda_vox1_testO $vox1_test_trials_O 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 ${exp}/scores.kaldi/scores_plda_vox1_testO $vox1_test_trials_O 2> /dev/null`
  echo "TEST O: EER: $eer%" | tee -a ${exp}/scores.kaldi/vox1_plda_result
  echo "minDCF(p-target=0.01): $mindcf1" | tee -a ${exp}/scores.kaldi/vox1_plda_result
  echo "minDCF(p-target=0.001): $mindcf2" | tee -a ${exp}/scores.kaldi/vox1_plda_result
fi

if [ $stage -le 3 ]; then
  $train_cmd ${exp}/scores.kaldi/log/vox1_testE_plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $train_emb_dir/plda - |" \
    "ark:ivector-subtract-global-mean $train_emb_dir/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | transform-vec $train_emb_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $train_emb_dir/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | transform-vec $train_emb_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$vox1_test_trials_E' | cut -d\  --fields=1,2 |" ${exp}/scores.kaldi/scores_plda_vox1_testE || exit 1;

  eer=`compute-eer <(local/prepare_for_eer.py $vox1_test_trials_E ${exp}/scores.kaldi/scores_plda_vox1_testE) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 ${exp}/scores.kaldi/scores_plda_vox1_testE $vox1_test_trials_E 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 ${exp}/scores.kaldi/scores_plda_vox1_testE $vox1_test_trials_E 2> /dev/null`
  echo "TEST E: EER: $eer%" | tee -a ${exp}/scores.kaldi/vox1_plda_result
  echo "minDCF(p-target=0.01): $mindcf1" | tee -a ${exp}/scores.kaldi/vox1_plda_result
  echo "minDCF(p-target=0.001): $mindcf2" | tee -a ${exp}/scores.kaldi/vox1_plda_result
fi

if [ $stage -le 4 ]; then
  $train_cmd ${exp}/scores.kaldi/log/vox1_testH_plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $train_emb_dir/plda - |" \
    "ark:ivector-subtract-global-mean $train_emb_dir/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | transform-vec $train_emb_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $train_emb_dir/mean.vec scp:$eval_emb_dir/xvector.scp ark:- | transform-vec $train_emb_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$vox1_test_trials_H' | cut -d\  --fields=1,2 |" ${exp}/scores.kaldi/scores_plda_vox1_testH || exit 1;

  eer=`compute-eer <(local/prepare_for_eer.py $vox1_test_trials_H ${exp}/scores.kaldi/scores_plda_vox1_testH) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 ${exp}/scores.kaldi/scores_plda_vox1_testH $vox1_test_trials_H 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 ${exp}/scores.kaldi/scores_plda_vox1_testH $vox1_test_trials_H 2> /dev/null`
  echo "TEST H: EER: $eer%" | tee -a ${exp}/scores.kaldi/vox1_plda_result
  echo "minDCF(p-target=0.01): $mindcf1" | tee -a ${exp}/scores.kaldi/vox1_plda_result
  echo "minDCF(p-target=0.001): $mindcf2" | tee -a ${exp}/scores.kaldi/vox1_plda_result
fi
