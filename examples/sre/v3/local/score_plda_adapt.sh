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
exp_dir=exp/ResNet34-TSTP-emb256-fbank40-num_frms200-aug0.6-spFalse-saFalse-Softmax-SGD-epoch10/
data=data
trials="${data}/sre16/eval/trials ${data}/sre16/eval/trials_tgl ${data}/sre16/eval/trials_yue"
aug_plda_data=0

enroll_scp=sre16/eval/enrollment/xvector.scp
test_scp=sre16/eval/test/xvector.scp
indomain_scp=sre16/major/xvector.scp    # For adaptation
utt2spk=data/sre16/eval/enrollment/utt2spk
preprocessing_path=${exp_dir}/embd_proc_sre16_major.pkl

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

if [ $aug_plda_data = 0 ];then
    sre_plda_data=sre
else
    sre_plda_data=sre_aug
fi

preproc_name=$(basename $preprocessing_path .pkl)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
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

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "adapt the plda model ..."

  indomain_scp=$(echo $indomain_scp | sed "s:\.scp:_proc_$preproc_name\.scp:")

  python wespeaker/bin/adapt_plda.py \
    -mo ${exp_dir}/plda \
    -ma ${exp_dir}/plda_adapt \
    -ad ${exp_dir}/embeddings/$indomain_scp \
    -ws 0.75 \
    -as 0.25
  echo "plda adapted finished"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "apply plda scoring ..."

  enroll_scp=$(echo $enroll_scp | sed "s:\.scp:_proc_$preproc_name\.scp:")
  test_scp=$(echo $test_scp | sed "s:\.scp:_proc_$preproc_name\.scp:")

  mkdir -p ${exp_dir}/scores
  for x in $(echo $trials | tr "," " "); do
    xx=$(basename  $x)
    echo "scoring on " $x
    python wespeaker/bin/eval_plda.py \
      --enroll_scp_path ${exp_dir}/embeddings/$enroll_scp \
      --test_scp_path ${exp_dir}/embeddings/$test_scp \
      --utt2spk $utt2spk \
      --trial ${x} \
      --score_path ${exp_dir}/scores/${xx}.proc_${preproc_name}_plda_adapt.score \
      --model_path ${exp_dir}/plda_adapt
  done
fi
#--indomain_scp ${exp_dir}/embeddings/$indomain_scp \ Note: This option was used before the new code for preprocessing.
# With this code, all preprocessing takes place in the preprocessing chain. So we don't include it in the above code anymore.

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "compute metrics (EER/minDCF) ..."
    scores_dir=${exp_dir}/scores
    for x in $(echo $trials | tr "," " "); do
        xx=$(basename  $x)
        python wespeaker/bin/compute_metrics.py \
            --p_target 0.01 \
            --c_fa 1 \
            --c_miss 1 \
            ${scores_dir}/${xx}.proc_${preproc_name}_plda_adapt.score \
            2>&1 | tee ${scores_dir}/${xx}.proc_${preproc_name}_plda_adapt.result
           #2>&1 | tee -a ${scores_dir}/${xx}_plda_adapt_result

        echo "compute DET curve ..."
        python wespeaker/bin/compute_det.py \
            ${scores_dir}/${xx}.proc_${preproc_name}_plda_adapt.score
    done
fi



