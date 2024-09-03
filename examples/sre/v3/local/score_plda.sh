#!/bin/bash

# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)
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
exp_dir="exp/ResNet34-TSTP-emb256-fbank40-num_frms200-aug0.6-spFalse-saFalse-Softmax-SGD-epoch10/"
data=data
trials="${data}/sre16/eval/trials ${data}/sre16/eval/trials_tgl ${data}/sre16/eval/trials_yue"
aug_plda_data=0

enroll_scp=sre16/eval/enrollment/xvector.scp
test_scp=sre16/eval/test/xvector.scp
utt2spk=data/sre16/eval/enrollment/utt2spk
preprocessing_chain='length-norm'
preprocessing_path="${exp_dir}/embd_proc.pkl"

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

if [ $aug_plda_data = 0 ];then
    sre_plda_data=cts
else
    sre_plda_data=cts_aug
fi

echo "preprocessing_path $preprocessing_path"
preproc_name=$(basename $preprocessing_path .pkl)
echo "preproc_name $preproc_name"



# Kaldi PLDA cts_aug, cts_aug mean, speaker mean last, no lnorm in PLDA
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preparing preprocessing chain for backend "
    python wespeaker/bin/prep_embd_proc.py \
    --chain "$preprocessing_chain" \
    --path $preprocessing_path
  echo "Backend preprocessor prepared"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Applying preprocessing on PLDA training data."
    python wespeaker/bin/apply_embd_proc.py \
    --path $preprocessing_path \
    --input ${exp_dir}/embeddings/${sre_plda_data}/xvector.scp \
    --output ${exp_dir}/embeddings/${sre_plda_data}/xvector_proc_$preproc_name.ark,scp
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "train the plda model ..."
  python wespeaker/bin/train_plda.py \
    --exp_dir ${exp_dir} \
    --scp_path ${exp_dir}/embeddings/${sre_plda_data}/xvector_proc_$preproc_name.scp \
    --utt2spk ${data}/${sre_plda_data}/utt2spk \
    --indim 100 \
    --iter 10
  echo "plda training finished"
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Applying preprocessing on evaluation and adaptation data."
    for x in $enroll_scp $test_scp $indomain_scp;do
        #new_x=$(echo $x | sed "s:\.scp:_proc\.ark,scp:")
        new_x=$(echo $x | sed "s:\.scp:_proc_$preproc_name\.ark,scp:")
        echo "Processing in: $x"
        echo "Processing out: $new_x"
        python wespeaker/bin/apply_embd_proc.py \
            --path $preprocessing_path \
            --input ${exp_dir}/embeddings/$x \
            --output ${exp_dir}/embeddings/$new_x
    done
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "apply plda scoring ..."
  mkdir -p ${exp_dir}/scores

  enroll_scp=$(echo $enroll_scp | sed "s:\.scp:_proc_$preproc_name\.scp:")
  test_scp=$(echo $test_scp | sed "s:\.scp:_proc_$preproc_name\.scp:")

  for x in $(echo $trials | tr "," " "); do
    xx=$(basename  $x)
    echo "scoring on " $x
    python wespeaker/bin/eval_plda.py \
      --enroll_scp_path ${exp_dir}/embeddings/$enroll_scp \
      --test_scp_path ${exp_dir}/embeddings/$test_scp \
      --utt2spk $utt2spk \
      --trial ${x} \
      --score_path ${exp_dir}/scores/${xx}.proc_${preproc_name}_plda.score \
      --model_path ${exp_dir}/plda
  done
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "compute metrics (EER/minDCF) ..."
    scores_dir=${exp_dir}/scores
    for x in $(echo $trials | tr "," " "); do
        xx=$(basename  $x)
        python wespeaker/bin/compute_metrics.py \
            --p_target 0.01 \
            --c_fa 1 \
            --c_miss 1 \
            ${scores_dir}/${xx}.proc_${preproc_name}_plda.score \
            2>&1 | tee ${scores_dir}/${xx}.proc_${preproc_name}_plda.result
            # 2>&1 | tee -a ${scores_dir}/${xx}_plda_result

        echo "compute DET curve ..."
        python wespeaker/bin/compute_det.py \
            ${scores_dir}/${xx}.proc_${preproc_name}_plda.score
    done
fi
