#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

exp=nothing
eval_name=airpods2_iphoneX
trials=/dockerdata/hongjiwang/sid/data/${eval_name}/trial.kaldi

. utils/parse_options.sh

emb_dir=$exp/embedding

eval_emb_dir=$emb_dir/$eval_name


mkdir -p $exp/scores


$train_cmd ${exp}/scores/log/${eval_name}_scoring.log \
ivector-compute-dot-products  "cat '$trials' | cut -d\  --fields=1,2 |" \
"ark:copy-vector scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
"ark:copy-vector scp:$eval_emb_dir/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
${exp}/scores/scores_cos_${eval_name} || exit 1;

eer=`compute-eer <(local/prepare_for_eer.py $trials ${exp}/scores/scores_cos_${eval_name}) 2> /dev/null`
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 ${exp}/scores/scores_cos_${eval_name} $trials 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 ${exp}/scores/scores_cos_${eval_name} $trials 2> /dev/null`
echo "Test ${eval_emb_dir}: EER: $eer%" | tee ${exp}/scores/${eval_name}_cos_result
echo "minDCF(p-target=0.01): $mindcf1" | tee -a ${exp}/scores/${eval_name}_cos_result
echo "minDCF(p-target=0.001): $mindcf2" | tee -a ${exp}/scores/${eval_name}_cos_result

