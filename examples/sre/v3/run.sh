#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#           2024 Johan Rohdin (rohdin@fit.vutbr.cz)

. ./path.sh || exit 1

# Stages
#  1. Data preparation
#  2. Shard / raw list creation
#  3. Training
#  4. Model averaging, embedding extraction
#  5. Export model
#  6. Cosine scoring using cts_aug, sre16_major, sre18_dev_unlabeled for mean subtraction but no other embedding processing
#  7. PLDA scoring, including length-norm, lda and subtraction of the above mentioned sets. See details at the stage.
#  8. Adapted PLDA scoring. Same embedding processing as above.
#  9. Cosine scoring with same embedding processing as above.
# 10. Summarization of results.

stage=1
stop_stage=1

HOST_NODE_ADDR="localhost:29400"
num_nodes=1
job_id=2024

data=data
data_type="shard"  # shard/raw

# whether augment the PLDA data
aug_plda_data=1

config=conf/resnet.yaml
exp_dir=exp/ResNet34-TSTP-emb256-fbank64-num_frms200-aug0.6-spFalse-saFalse-Softmax-SGD-epoch10

# gpus="[0,1]" # For slurm, just specify this according to the number of GPUs you have.
num_gpus_train=2  # If this variable is defined, safe_gpu will be used to select the free GPUs.
                  # If so, it will override whatever may have been specified in gpus="[x,...]
                  # Typically, you would want to use this option for SGE.
                  # If this variable is not set, or set to '', the script will assume that
                  # the GPUs to use are specified in the variable "gpus" as above.

num_gpus_extract=4 # We may want to use a different value for extraction.

num_avg=10
checkpoint=


. tools/parse_options.sh || exit 1

############################################################################################
# The names of various lists are not consistent across sets. Therefore we need some mappings.

# Different sets may use different backend adaptation sets, therefore we need several trial
# lists. Using "," instead of space as separator is a bit ugly but it seems parse_options.sh
# cannot process an argument with space properly.
declare -A trials=( ["sre16_eval"]='data/sre16/eval/trials,data/sre16/eval/trials_yue,data/sre16/eval/trials_tgl'
    ["sre18_dev"]="data/sre18/dev/sre18_dev_trials"
    ["sre18_eval"]="data/sre18/eval/sre18_eval_trials"
    ["sre21_dev"]="data/sre21/dev/sre21_dev_trials"
    ["sre21_eval"]="data/sre21/eval/sre21_eval_trials" )

declare -A enr_scp=( ["sre16_eval"]='sre16/eval/enrollment/xvector.scp'
    ["sre18_dev"]="sre18/dev/enrollment/xvector.scp"
    ["sre18_eval"]="sre18/eval/enrollment/xvector.scp"
    ["sre21_dev"]="sre21/dev/enrollment/xvector.scp"
    ["sre21_eval"]="sre21/eval/enrollment/xvector.scp" )

declare -A test_scp=( ["sre16_eval"]='sre16/eval/test/xvector.scp'
    ["sre18_dev"]="sre18/dev/test/xvector.scp"
    ["sre18_eval"]="sre18/eval/test/xvector.scp"
    ["sre21_dev"]="sre21/dev/test/xvector.scp"
    ["sre21_eval"]="sre21/eval/test/xvector.scp" )

declare -A utt2mdl=( ["sre16_eval"]='data/sre16/eval/enrollment/utt2spk'
    ["sre18_dev"]="data/sre18/dev/enrollment/utt2mdl_id"
    ["sre18_eval"]="data/sre18/eval/enrollment/utt2mdl_id"
    ["sre21_dev"]="data/sre21/dev/enrollment/utt2mdl_id"
    ["sre21_eval"]="data/sre21/eval/enrollment/utt2mdl_id" )

declare -A mdl2utt=( ["sre16_eval"]='data/sre16/eval/enrollment/spk2utt'
    ["sre18_dev"]="data/sre18/dev/enrollment/mdl_id2utt"
    ["sre18_eval"]="data/sre18/eval/enrollment/mdl_id2utt"
    ["sre21_dev"]="data/sre21/dev/enrollment/mdl_id2utt"
    ["sre21_eval"]="data/sre21/eval/enrollment/mdl_id2utt" )

declare -A xvectors=( ["sre16_eval"]="sre16/eval/xvector.scp"
    ["sre18_dev"]="sre18/dev/xvector.scp"
    ["sre18_eval"]="sre18/eval/xvector.scp"
    ["sre21_dev"]="sre21/dev/xvector.scp"
    ["sre21_eval"]="sre21/eval/xvector.scp" )
############################################################################################


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."


  ######################################################################################
  ### Test sets. Please specify paths
  # SRE16 should be prepared by the Kaldi recipe and the path should be specified here:
  #sre_data_dir=/mnt/matylda4/burget/kaldi-trunk/kaldi/egs/sre16/v2/data/
  # Will be used by ./local/prepare_data.sh below. (only wav.scp, utt2spk and spk2utt files are needed.)
  sre16_unlab_dir=/mnt/matylda2/data/NIST/sre16/LDC2016E46_SRE16_Call_My_Net_Training_Data
  sre16_evalset_dir=/mnt/matylda2/data/NIST/sre16/R149_0_1
  # Eval keys are not in the above directory since they were distributed after the evaluation.
  sre16_evalset_keys=/mnt/matylda2/data/NIST/sre16/download/sre16_evaluation_key.tar.bz2

  # SRE18
  sre18_devset_dir=/mnt/matylda2/data/NIST/sre18/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set
  sre18_evalset_dir=/mnt/matylda2/data/LDC/LDC2018E51_2018_NIST_Speaker_Recognition_Evaluation_Test_Set/
  # Eval keys are not in the above directory since they were distributed after the evaluation.
  sre18_evalset_keys=/mnt/matylda2/data/NIST/sre18/LDC2018E51_eval_segment_key.tbz2

  # SRE21
  sre21_devset_dir=/mnt/matylda2/data/LDC/LDC2021E09_sre21_dev_set/
  sre21_evalset_dir=/mnt/matylda2/data/LDC/LDC2021E10_sre21_eval_set/
  # Eval keys are not in the above directory since they were distributed after the evaluation.
  sre21_evalset_keys=/mnt/matylda2/data/NIST/sre21/download/sre21_test_key.tgz


  ######################################################################################
  ### Training sets
  # CTS
  cts_superset_dir=/mnt/matylda2/data/LDC/LDC2021E08_SRE-CTS-Superset/

  # VoxCeleb
  voxceleb_dir="/mnt/matylda6/rohdin/expts/wespeaker/wespeaker/examples/voxceleb/v2/data/"

  # This script is based on ../v2/local/prepare_data.sh
  # Copies SRE16 relevant files, extracts VAD for all files, does some pruning of the training set.
  ./local/prepare_data.sh --stage 1 --stop_stage 10 --data ${data} \
                          --sre16_unlab_dir ${sre16_unlab_dir} --sre16_evalset_dir ${sre16_evalset_dir} --sre16_evalset_keys ${sre16_evalset_keys} \
                          --sre18_devset_dir ${sre18_devset_dir} --sre18_evalset_dir ${sre18_evalset_dir} --sre18_evalset_keys ${sre18_evalset_keys} \
                          --sre21_devset_dir ${sre21_devset_dir} --sre21_evalset_dir ${sre21_evalset_dir} --sre21_evalset_keys ${sre21_evalset_keys} \
                          --cts_superset_dir ${cts_superset_dir} --voxceleb_dir ${voxceleb_dir}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

  true && {
      echo "Convert train data to ${data_type}..."
      for dset in cts_vox; do
          python tools/make_shard_list.py --num_utts_per_shard 1000 \
              --num_threads 12 \
              --prefix shards \
              --shuffle \
              --vad_file ${data}/$dset/vad \
              ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
              ${data}/$dset/shards ${data}/$dset/shard.list
      done
  }

  true && {
      echo "Convert data for PLDA backend training and evaluation to raw format..."
      if [ $aug_plda_data = 0 ];then
          sre_plda_data=cts
      else
          sre_plda_data=cts_aug
      fi

      # Raw format for backend and evaluation data
      for dset in ${sre_plda_data} sre16/major sre16/eval/enrollment sre16/eval/test \
          sre18/dev/enrollment sre18/dev/test sre18/dev/unlabeled sre18/eval/enrollment sre18/eval/test \
          sre21/dev/enrollment sre21/dev/test sre21/eval/enrollment sre21/eval/test;do

          # The below requires utt2spk to be present. So create a "dummy" one if we don't have it.
          # This is for example the case with sre21 eval data.
          if [ ! -f $data/$dset/utt2spk ];then
              awk '{print $1 " unk"}' ${data}/${dset}/wav.scp > ${data}/${dset}/utt2spk
          fi

          python tools/make_raw_list.py --vad_file ${data}/$dset/vad \
              ${data}/$dset/wav.scp \
              ${data}/$dset/utt2spk ${data}/$dset/raw.list
      done
  }

  true && {
      # Convert all musan and rirs data to LMDB if they don't already exist.
      for x in rirs musan;do
          if [ ! -d $data/$x/lmdb ];then
              python tools/make_lmdb.py ${data}/$x/wav.scp ${data}/$x/lmdb
          fi
      done
  }

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  if [ ! -z $num_gpus_train ];then
      gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus_train | sed "s: ::g")
  else
      num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
      wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/cts_vox/${data_type}.list \
      --train_label ${data}/cts_vox/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

  false &&  {
      echo "Do model average ..."
      avg_model=$exp_dir/models/avg_model.pt
      python wespeaker/bin/average_model.py \
          --dst_model $avg_model \
          --src_path $exp_dir/models \
          --num ${num_avg}

      model_path=$avg_model
      if [[ $config == *repvgg*.yaml ]]; then
          echo "convert repvgg model ..."
          python wespeaker/models/convert_repvgg.py \
              --config $exp_dir/config.yaml \
              --load $avg_model \
              --save $exp_dir/models/convert_model.pt
          model_path=$exp_dir/models/convert_model.pt
      fi
  }

  avg_model=$exp_dir/models/avg_model.pt
  model_path=$avg_model

  echo "Extract embeddings ..."
  avg_model=$exp_dir/models/avg_model.pt
  model_path=$avg_model
  gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus_extract | sed "s: ::g" )
  echo $gpus
  local/extract_sre.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj $num_gpus_extract --gpus $gpus --data_type raw --data ${data} \
    --reverb_data ${data}/rirs/lmdb \
    --noise_data ${data}/musan/lmdb \
    --aug_plda_data ${aug_plda_data}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Export the final model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "### --- Score using Cosine Distance --- ###"

    # Use SRE16 unlabeled data for mean subraction
    echo "### --- Mean: SRE16 unlabeled ("SRE16 Major")  --- ###"
    true && {
    for dset in sre16_eval;do
        echo " * $dset"
        local/score.sh \
            --stage 1 --stop-stage 2 \
            --trials ${trials[$dset]} \
            --xvectors $exp_dir/embeddings/${xvectors[$dset]} \
            --cal_mean_dir ${exp_dir}/embeddings/sre16/major \
            --exp_dir $exp_dir
    done
    }

    # Use SRE18 unlabeled data for mean subraction
    echo "### --- Mean: SRE18 Unlabeled --- ###"
    true && {
    for dset in sre18_eval sre18_dev;do
        echo " * $dset"
        local/score.sh \
            --stage 1 --stop-stage 2 \
            --trials ${trials[$dset]} \
            --xvectors $exp_dir/embeddings/${xvectors[$dset]} \
            --cal_mean_dir ${exp_dir}/embeddings/sre18/dev/unlabeled \
            --exp_dir $exp_dir
    done
    }

    # Use backend training data for mean subraction
    echo "### --- Mean: SRE --- ###"
    true && {
    for dset in sre16_eval sre18_eval sre18_dev sre21_eval sre21_dev;do
        echo " * $dset"
        local/score.sh \
            --stage 1 --stop-stage 2 \
            --trials ${trials[$dset]} \
            --xvectors $exp_dir/embeddings/${xvectors[$dset]} \
            --cal_mean_dir ${exp_dir}/embeddings/cts_aug \
            --exp_dir $exp_dir
    done
    }

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "### --- Score with PLDA --- ###"
  echo "### --- Mean: PLDA training set (cts_aug) --- ###"

  # Here we specify the embedding preprocessing to be used before backend modelling/scoring.
  mean1_scp=${exp_dir}/embeddings/${sre_plda_data}/cts_aug/xvector.scp
  lda_scp=${exp_dir}/embeddings/${sre_plda_data}/cts_aug/xvector.scp
  utt2spk=${data}/cts_aug/utt2spk
  lda_dim=100
  preprocessing_chain="mean-subtract --scp $mean1_scp | length-norm | lda --scp $lda_scp --utt2spk $utt2spk --dim $lda_dim | length-norm"
  preprocessing_path_cts_aug=${exp_dir}/embd_proc_cts_aug.pkl

  # Run stage 1-6 here to train the embedding preprocessing chain and the PLDA model as well
  # as to evaluate SRE16 which is the default set to evaluate if no eval set is provided.
  true && {
   local/score_plda.sh \
    --stage 1 --stop-stage 6 \
    --data ${data} \
    --exp_dir $exp_dir \
    --aug_plda_data ${aug_plda_data} \
    --preprocessing_chain "$preprocessing_chain" \
    --preprocessing_path "$preprocessing_path_cts_aug"
  }
  # Score the other sets. We need only stage 4-6 for this.
  true && {
      for dset in sre18_eval sre18_dev sre21_eval sre21_dev;do
          local/score_plda.sh \
              --stage 4 --stop-stage 6 \
              --data ${data} \
              --exp_dir $exp_dir \
              --enroll_scp ${enr_scp[$dset]} \
              --test_scp ${test_scp[$dset]} \
              --aug_plda_data ${aug_plda_data} \
              --preprocessing_path "$preprocessing_path" \
              --preprocessing_path "$preprocessing_path_cts_aug" \
              --utt2spk ${utt2mdl[$dset]} \
              --trials ${trials[$dset]}
      done
  }

  # Score using SRE 16 unlab mean. We should not retrain the backend again, i.e. stage 2-3
  # but we do need to update the embedding preprocessing chain.
  mean1_scp=${exp_dir}/embeddings/sre16/major/xvector.scp
  new_link="mean-subtract --scp $mean1_scp "
  preprocessing_path_sre16_major=${exp_dir}/embd_proc_sre16_major.pkl

  # The following command replaces link 0 (cts_aug mean subtraction) with a new link (sre16 major mean subtraction)
  python wespeaker/bin/update_embd_proc.py --in_path $preprocessing_path_cts_aug --out_path $preprocessing_path_sre16_major --link_no_to_remove 0 --new_link "$new_link"

  echo "### --- Mean: SRE16 Major --- ###"
  true && {
  local/score_plda.sh \
    --stage 4 --stop-stage 6 \
    --data ${data} \
    --exp_dir $exp_dir \
    --preprocessing_path "$preprocessing_path_sre16_major"
  }

  # Similarly for SRE18
  mean1_scp=${exp_dir}/embeddings/sre18/dev/unlabeled/xvector.scp
  new_link="mean-subtract --scp $mean1_scp "
  preprocessing_path_sre18_unlab=${exp_dir}/embd_proc_sre18_dev_unlabeled.pkl

  python wespeaker/bin/update_embd_proc.py --in_path $preprocessing_path_cts_aug --out_path $preprocessing_path_sre18_unlab --link_no_to_remove 0 --new_link "$new_link"

  echo "### --- Mean: SRE18 Unlabeled --- ###"
  true && {
  for dset in sre18_eval sre18_dev;do
   local/score_plda.sh \
    --stage 4 --stop-stage 6 \
    --data ${data} \
    --exp_dir $exp_dir \
    --preprocessing_path "$preprocessing_path_sre18_unlab" \
    --enroll_scp ${enr_scp[$dset]} \
    --test_scp ${test_scp[$dset]} \
    --utt2spk ${utt2mdl[$dset]} \
    --trials ${trials[$dset]}
  done
  }
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then

  echo "Score with adapted PLDA ..."

  # --indomain_scp is by default sre16/major/xvector.scp in local/score_plda_adapt.sh.
  # It is used for adaptation. Note that in other recipes, indomain_scp is passed to
  # wespeaker/bin/eval_plda.py insdide local/score_plda_adapt.sh in which case it will
  # be used for mean subtraction before scoring. In this recipe, mean subtraction is,
  # however, part of the backend preprocessing chain and is therefore not used in
  # wespeaker/bin/eval_plda.py.

  echo "### --- Mean: SRE16 Major --- ###"
  true && {
  local/score_plda_adapt.sh \
      --stage 1 --stop-stage 4 \
      --data ${data} \
      --exp_dir $exp_dir \
      --preprocessing_path ${exp_dir}/embd_proc_sre16_major.pkl \
      --aug_plda_data ${aug_plda_data}
  }

  preprocessing_path_sre18_unlab=${exp_dir}/embd_proc_sre18_dev_unlabeled.pkl
  echo "### --- Mean: SRE18 Unlabeled --- ###"
  # Stage 1 is only needed to be run once per domain so we could have set stage 1-4 for
  # sre18_eval and stage 1,3,4 for sre18_dev but since stage 2 is very fast we keep it
  # in order to keep the script clean.
  true && {
  for dset in sre18_eval sre18_dev;do

      local/score_plda_adapt.sh \
          --stage 1 --stop-stage 4 \
          --data ${data} \
          --exp_dir $exp_dir \
          --aug_plda_data ${aug_plda_data} \
          --enroll_scp ${enr_scp[$dset]} \
          --test_scp ${test_scp[$dset]} \
          --preprocessing_path "$preprocessing_path_sre18_unlab" \
          --indomain_scp sre18/dev/unlabeled/xvector.scp \
          --utt2spk ${utt2mdl[$dset]} \
          --trials ${trials[$dset]}
  done
  }
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "### --- Score using Cosine Distance --- ###"

    # The preprocessed embeddings are already stored but we need to create the lists as
    # score.sh wants them. This is a bit messy and therefore kept in a separate script.
    ./local/create_preproc_embd_lists.sh $exp_dir

    # Note that cal_mean_dir should not be provided since the embedding preprocessing includes mean subtration.

    # Use SRE16 unlabeled data for mean subraction
    echo "### --- Mean: SRE16 unlabeled ("SRE16 Major")  --- ###"
    true && {
    preproc_name=embd_proc_sre16_major
    for dset in sre16_eval;do
        # The xvector list for the relevant preprocessing chain.
        new_xvectors=$(echo $exp_dir/embeddings/${xvectors[$dset]} | sed "s:\.scp:_proc_$preproc_name\.scp:")
        echo " * $new_xvectors"
        local/score.sh \
            --stage 1 --stop-stage 2 \
            --trials ${trials[$dset]} \
            --xvectors $new_xvectors \
            --exp_dir $exp_dir
    done

    }

    # Use SRE18 unlabeled data for mean subraction
    echo "### --- Mean: SRE18 Unlabeled --- ###"
    true && {
    preproc_name=embd_proc_sre18_dev_unlabeled
    for dset in sre18_eval sre18_dev;do
        new_xvectors=$(echo $exp_dir/embeddings/${xvectors[$dset]} | sed "s:\.scp:_proc_$preproc_name\.scp:")
        echo " * $new_xvectors"
        local/score.sh \
            --stage 1 --stop-stage 2 \
            --trials ${trials[$dset]} \
            --xvectors $new_xvectors \
            --exp_dir $exp_dir
    done
    }

    # Use backend training data for mean subraction
    echo "### --- Mean: SRE --- ###"
    true && {
    preproc_name=embd_proc_cts_aug
    for dset in sre16_eval sre18_eval sre18_dev sre21_eval sre21_dev;do
        new_xvectors=$(echo $exp_dir/embeddings/${xvectors[$dset]} | sed "s:\.scp:_proc_$preproc_name\.scp:")
        echo " * $new_xvectors"
        local/score.sh \
            --stage 1 --stop-stage 2 \
            --trials ${trials[$dset]} \
            --xvectors $new_xvectors \
            --exp_dir $exp_dir
    done
    }

fi



if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    # Summarize results
    echo ""
    echo "----------------------------------------------------"
    echo "### --- Summary of results (EER / minDCF0.01)--- ###"
    echo "----------------------------------------------------"
    # Make the header
    eval_data='system'
    for dset in sre16_eval sre18_dev sre18_eval sre21_dev sre21_eval;do
        for x in $(echo ${trials[$dset]} | tr "," " "); do
            xx=$(basename  $x)
            eval_data="$eval_data, $xx  "
        done
    done
    echo $eval_data > results_summary.txt
    # Collect the results
    for sys in mean_cts_aug_cos mean_sre16_major_cos mean_sre18_dev_unlabeled_cos \
               proc_embd_proc_cts_aug_cos proc_embd_proc_sre16_major_cos proc_embd_proc_sre18_dev_unlabeled_cos \
               proc_embd_proc_cts_aug_plda proc_embd_proc_sre16_major_plda proc_embd_proc_sre18_dev_unlabeled_plda \
               proc_embd_proc_sre16_major_plda_adapt  proc_embd_proc_sre18_dev_unlabeled_plda_adapt;do
        res="$sys,"
        for dset in sre16_eval sre18_dev sre18_eval sre21_dev sre21_eval;do
            for x in $(echo ${trials[$dset]} | tr "," " "); do
                xx=$(basename  $x)
                eval_data="$eval_data $xx  "
                if [ -e ${exp_dir}/scores/${xx}.${sys}.result ];then
                    res="$res $(grep EER ${exp_dir}/scores/${xx}.${sys}.result | sed 's:.* = ::')"
                    res="$res / $(grep minDCF ${exp_dir}/scores/${xx}.${sys}.result | sed 's:.* = ::'),"
                else
                    res="$res - -,"
                fi
            done
        done
        echo -e $res >> results_summary.txt
    done
    column -t -s"," results_summary.txt
    echo ""
    echo "-------------------------------------------------------"
    echo "### --- CSV for copy-paste to google sheet etc. --- ###"
    echo "-------------------------------------------------------"
    tail -n+2 results_summary.txt | sed "s:/:,:g" | sed "s: :,:g"| sed -r "s:,+:,:g"


fi
