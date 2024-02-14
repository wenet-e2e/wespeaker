#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#           2024 Johan Rohdin (rohdin@fit.vutbr.cz)

. ./path.sh || exit 1

stage=1
stop_stage=1

# the sre data should be prepared in kaldi format and stored in the following directory


data=data
data_type="shard"  # shard/raw
# whether augment the PLDA data
aug_plda_data=0

config=conf/resnet.yaml
exp_dir=exp/ResNet34-TSTP-emb256-fbank40-num_frms200-aug0.6-spFalse-saFalse-Softmax-SGD-epoch10

# gpus="[0,1]" # Is set below at the relevant stage
# num_gpus=2

num_avg=10
checkpoint=


. tools/parse_options.sh || exit 1


# Different set may use different backend training sets, therefore we need several trial lists
# Using "," instead of space as separator is a bit ugly but it seems parse_options,sh connot
# process an argument with space properly.
declare -A trials=( ["sre16_eval"]='data/trials/trials,data/trials/trials_tgl,data/trials/trials_yue'
    ["sre18_dev"]="data/sre18/dev/sre18_dev.trials"
    ["sre18_eval"]="data/sre18/eval/sre18_eval.trials"
    ["sre21_dev"]="data/sre21/dev/sre21_dev.trials"
    ["sre21_eval"]="data/sre21/eval/sre21_eval.trials" )

declare -A enr_scp=( ["sre16_eval"]='sre16_eval_enroll/xvector.scp'
    ["sre18_dev"]="sre18/dev/enrollment/xvector.scp"
    ["sre18_eval"]="sre18/eval/enrollment/xvector.scp"
    ["sre21_dev"]="sre21/dev/enrollment/xvector.scp"
    ["sre21_eval"]="sre21/eval/enrollment/xvector.scp" )

declare -A test_scp=( ["sre16_eval"]='sre16_eval_test/xvector.scp'
    ["sre18_dev"]="sre18/dev/test/xvector.scp"
    ["sre18_eval"]="sre18/eval/test/xvector.scp"
    ["sre21_dev"]="sre21/dev/test/xvector.scp"
    ["sre21_eval"]="sre21/eval/test/xvector.scp" )

declare -A utt2mdl=( ["sre16_eval"]='data/sre16_eval_enroll/utt2spk'
    ["sre18_dev"]="data/sre18/dev/enrollment/utt2mdl_id"
    ["sre18_eval"]="data/sre18/eval/enrollment/utt2mdl_id"
    ["sre21_dev"]="data/sre21/dev/enrollment/utt2mdl_id"
    ["sre21_eval"]="data/sre21/eval/enrollment/utt2mdl_id" )

declare -A xvectors=( ["sre16_eval"]="sre16_eval/xvector.scp"
    ["sre18_dev"]="sre18/dev/xvector.scp"
    ["sre18_eval"]="sre18/eval/xvector.scp"
    ["sre21_dev"]="sre21/dev/xvector.scp"
    ["sre21_eval"]="sre21/eval/xvector.scp" )



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."

  # SRE16 should be prepared by the Kaldi recipe and the path should be specified here:
  sre_data_dir=/mnt/matylda4/burget/kaldi-trunk/kaldi/egs/sre16/v2/data/
  # Will be used by ./local/prepare_data.sh below. (only wav.scp, utt2spk and spk2utt files are needed.)

  # SRE18
  false && {
      sre18_devset_dir=/mnt/matylda2/data/NIST/sre18/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set
      sre18_evalset_dir=/mnt/matylda2/data/LDC/LDC2018E51_2018_NIST_Speaker_Recognition_Evaluation_Test_Set/
      # Eval keys are not in the above directory since they were distributed after the evaluation.
      sre18_evalset_keys=/mnt/matylda2/data/NIST/sre18/LDC2018E51_eval_segment_key.tbz2
      
      # Remove stage option?
      local/prepare_sre18.sh --stage 1 --stop_stage 1 --sre18_dev_dir $sre18_devset_dir --sre18_eval_dir $sre18_evalset_dir --sre18_eval_keys_file $sre18_evalset_keys --data_dir $data/sre18
  }

  # SRE21
  false && {
      sre21_devset_dir=/mnt/matylda2/data/LDC/LDC2021E09_sre21_dev_set/
      sre21_evalset_dir=/mnt/matylda2/data/LDC/LDC2021E10_sre21_eval_set/
      # Eval keys are not in the above directory since they were distributed after the evaluation.
      sre21_evalset_keys=/mnt/matylda2/data/NIST/sre21/download/sre21_test_key.tgz
      local/prepare_sre21.sh --stage 1 --stop_stage 1 --sre21_dev_dir $sre21_devset_dir --sre21_eval_dir $sre21_evalset_dir --sre21_eval_keys_file $sre21_evalset_keys --data_dir $data/sre21
  }

  # Prepare CTS
  false && {
      cts_superset_dir=/mnt/matylda2/data/LDC/LDC2021E08_SRE-CTS-Superset/
      local/prepare_cts_superset.sh --cts_superset_dir $cts_superset_dir --data_cts $data/cts --wav_dir `pwd`/wav/cts
  
      # Only mixer data. Used for backend training. 
      awk -F"\t" '{if($7 == "mx3" || $7 ==  "mx45" || $7 == "mx6"){print $0}  }' ${cts_superset_dir}/docs/cts_superset_segment_key.tsv \
	  > data/cts_superset_segment_key_mx3456.tsv
      cut -f 1  data/cts_superset_segment_key_mx3456.tsv | sed s:\\.sph$:: > data/mx_3456.list  
  }


  ######################################################################################
  ### VoxCeleb
  ######################################################################################
  # We are using all of VoxCeleb 1 and the training (aka "development") part of VoxCeleb 2.
  # (The test part of VoxCeleb 2) may have some overlap with VoxCeleb 1. See 
  # https://www.robots.ox.ac.uk/~vgg/publications/2019/Nagrani19/nagrani19.pdf, Table 4.)

  vox_dir="/mnt/matylda6/rohdin/expts/wespeaker/wespeaker/examples/voxceleb/v2/data/"
  #vox_dir=""
  false && {

      if [[ $vox_dir == "" ]];then
          echo "Preparing Voxceleb, rirs and Musan"
	  vox_dir=${data}_vox
	  mkdir ${vox_dir}
          local/prepare_vox<.sh --stage 1 --stop_stage 4 --data ${data}_vox
      fi

      if [[ ! -d $vox_dir/vox1  ||  ! -d $vox_dir/vox2_dev ]];then
          echo "ERROR: Problem with Voxceleb data directory."
          exit 1
      fi
      
      # TODO: It would be nice to have a script that applies a general SOX command to 
      #       data instead of the below that does specifically downsampling followed 
      #       by application of GSM encoding.
      #       Also, the script creates the new wav files instead of adding a piped
      #       command in the scp. This was because at the time of writing the script,
      #       other wesaper tool did not support piped commands in wav.scp.
      tools/downsample_and_apply_gsm.sh --src_dir $vox_dir/vox1 --dest_dir $data/vox1_gsmfr --rate 8k --wav_dir `pwd`/wav/vox1_gsmfr --remove_prefix_wav $vox_dir
      tools/downsample_and_apply_gsm.sh --src_dir $vox_dir/vox2_dev --dest_dir $data/vox2_dev_gsmfr --rate 8k --wav_dir `pwd`/wav/vox2_dev_gsmfr --remove_prefix_wav $vox_dir
      
      # Combine all Voxceleb data
      utils/combine_data.sh data/vox_gsmfr data/vox1_gsmfr/ data/vox2_dev_gsmfr/      
  }
  ######################################################################################

  true && {
      # This script is taken from ../v2/ where it is used to prepare other training datasets plus sre16 
      # eval sets. 
      # Copies SRE16 relevant files, extracts VAD for all files, does some pruning of the training set.
      ./local/prepare_data.sh --stage 3 --stop_stage 3 --data ${data} --sre_data_dir ${sre_data_dir}    
  }

  #./utils/combine_data.sh data/cts_vox data/cts/ data/vox_gsmfr 

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    
  false && {
      cp -r data/cts cts_tmp
      ./utils/fix_data_dir.sh cts_tmp/
      mv cts_tmp/vad cts_tmp/vad.org
      LC_ALL=C sort -k2 cts_tmp/vad.org > cts_tmp/vad
  }

  false && {  
  echo "Convert train data to ${data_type}..."
  for dset in cts; do
  #for dset in ../cts_tmp; do
        python tools/make_shard_list.py --num_utts_per_shard 1000 \
            --num_threads 12 \
            --prefix shards \
            --shuffle \
            --vad_file ${data}/$dset/vad \
            ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
            ${data}/$dset/shards ${data}/$dset/shard.list
  done
  }

  echo "Convert data for PLDA backend training and evaluation to raw format..."
  if [ $aug_plda_data = 0 ];then
      sre_plda_data=mx_3456
  else
      sre_plda_data=mx_3456_aug
  fi
  
  true && {
  for dset in ${sre_plda_data}; do

  # These were not ran since data from ../v2 which already had raw.list was used.
  #for dset in sre16_major sre16_eval_enroll sre16_eval_test sre18/dev/enrollment sre18/dev/test sre18/dev/unlabeled sre18/eval/enrollment sre18/eval/test sre21/dev/enrollment sre21/dev/test sre21/eval/enrollment sre21/eval/test; do

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

  false && {
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
  }

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
 gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  #gpus="[0,1]"
  echo $gpus
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/cts/${data_type}.list \
      --train_label ${data}/cts/utt2spk \
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
  num_gpus=4 #1
  gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g" )
  echo $gpus
  local/extract_sre.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj $num_gpus --gpus $gpus --data_type raw --data ${data} \
    --reverb_data ${data}/rirs/lmdb \
    --noise_data ${data}/musan/lmdb \
    --aug_plda_data ${aug_plda_data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Score using Cosine Distance..."

    # Add so that score file name contains which mean was used for subtraction otherwise 
    # make sure to always run from stage 1.

    # Use SRE16 unlabeled data for mean subraction
    echo "### --- Mean: SRE16 unlabeled ("SRE16 Major")  --- ###" 
    true && {
    for dset in sre16_eval;do
	echo " * $dset"
	local/score.sh \
	    --stage 1 --stop-stage 2 \
	    --trials ${trials[$dset]} \
            --xvectors $exp_dir/embeddings/${xvectors[$dset]} \
            --cal_mean_dir ${exp_dir}/embeddings/sre16_major \
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
            --cal_mean_dir ${exp_dir}/embeddings/mx_3456 \
            --exp_dir $exp_dir 
    done
    }

fi

false && {
# This is a hack for BUT since these files are present in the scp but not in utt2spk which causes an error.
grep -v SANITY $exp_dir/embeddings/sre21/eval/enrollment/xvector.scp > $exp_dir/embeddings/sre21/eval/enrollment/xvector2.scp
grep -v SANITY $exp_dir/embeddings/sre21/dev/enrollment/xvector.scp > $exp_dir/embeddings/sre21/dev/enrollment/xvector2.scp
grep -v SANITY $exp_dir/embeddings/sre18/eval/enrollment/xvector.scp > $exp_dir/embeddings/sre18/eval/enrollment/xvector2.scp
grep -v SANITY $exp_dir/embeddings/sre18/dev/enrollment/xvector.scp > $exp_dir/embeddings/sre18/dev/enrollment/xvector2.scp
enr_scp["sre18_dev"]="sre18/dev/enrollment/xvector2.scp"
enr_scp["sre18_eval"]="sre18/eval/enrollment/xvector2.scp"
enr_scp["sre21_dev"]="sre21/dev/enrollment/xvector2.scp"
enr_scp["sre21_eval"]="sre21/eval/enrollment/xvector2.scp"
}

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score with PLDA ..."
  
  # Run only stage 1 here to train the model. (If more stages are run, it wil score SRE16
  # using sre16 unlabeled mean for mean subtraction)
  echo "### --- Mean: SRE16 unlabeled ("SRE16 Major")  --- ###" 
  false && {
   local/score_plda.sh \
    --stage 1 --stop-stage 3 \
    --data ${data} \
    --exp_dir $exp_dir \
    --aug_plda_data ${aug_plda_data} 
  }

  # Score using SRE 18 unlab mean. (No need to train the model again, e.g. stage 1)
  echo "### --- Mean: SRE18 Unlabeled --- ###" 
  false && {
  for dset in sre18_eval sre18_dev;do
   local/score_plda.sh \
    --stage 2 --stop-stage 3 \
    --data ${data} \
    --exp_dir $exp_dir \
    --aug_plda_data ${aug_plda_data} \
    --enroll_scp ${enr_scp[$dset]} \
    --test_scp ${test_scp[$dset]} \
    --indomain_scp sre18/dev/unlabeled/xvector.scp \
    --utt2spk ${utt2mdl[$dset]} \
    --trials ${trials[$dset]}
  done 
  }

  # Score using SRE mean.
  echo "### --- Mean: SRE --- ###" 
  true && {
  for dset in sre16_eval sre18_eval sre18_dev sre21_eval sre21_dev;do
   local/score_plda.sh \
    --stage 2 --stop-stage 3 \
    --data ${data} \
    --exp_dir $exp_dir \
    --aug_plda_data ${aug_plda_data} \
    --enroll_scp ${enr_scp[$dset]} \
    --test_scp ${test_scp[$dset]} \
    --indomain_scp mx_3456/xvector.scp \
    --utt2spk ${utt2mdl[$dset]} \
    --trials ${trials[$dset]}
  done 
  }

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score with adapted PLDA ..."
  true && {
      local/score_plda_adapt.sh \
	  --stage 1 --stop-stage 3 \
	  --data ${data} \
	  --exp_dir $exp_dir \
	  --aug_plda_data ${aug_plda_data}
  }


  true && { 
      # Stage 1 is only needed to be run once, but since it is very fast we keep it 
      # in the loop for keeping things clean.
      for dset in sre18_eval sre18_dev;do
	  local/score_plda_adapt.sh \
	      --stage 1 --stop-stage 3 \
	      --data ${data} \
	      --exp_dir $exp_dir \
	      --aug_plda_data ${aug_plda_data} \
	      --enroll_scp ${enr_scp[$dset]} \
	      --test_scp ${test_scp[$dset]} \
	      --indomain_scp sre18/dev/unlabeled/xvector.scp \
	      --utt2spk ${utt2mdl[$dset]} \
	      --trials ${trials[$dset]}
      done
  }
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi
