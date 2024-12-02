#!/bin/bash

# Copyright 2024 Yiyang Zhao (zhaoyy22@mails.tsinghua.edu.cn)
#           2024 Hongji Wang (jijijiang77@gmail.com)

. ./path.sh || exit 1

stage=-1
stop_stage=-1

HOST_NODE_ADDR="localhost:29400"
num_nodes=1
job_id=2024

data=data
data_type="raw"  # shard/raw
model=whisper_PMFA_large_v2

exp_dir=exp/Whisper_PMFA_large_v2_voxceleb1_mel_5s
gpus="[0,1]"
num_avg=1
checkpoint=

trials="vox1_O_cleaned.kaldi"

score_norm_method="asnorm"  # asnorm/snorm
top_n=300

. tools/parse_options.sh || exit 1
if ! pip show openai-whisper > /dev/null 2>&1; then
    pip install openai-whisper==20231117
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Preparing datasets ..."
  ./local/prepare_data.sh --stage 1 --stop_stage 3 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox1_dev vox1_test; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training with frozen whisper parameter..."
  config=conf/whisper_PMFA_stage0.yaml
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox1_dev/${data_type}.list \
      --train_label ${data}/vox1_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      --model ${model}
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Start training with all parameter..."

  if [ -f ${exp_dir}/"config.yaml" ]; then
    mv ${exp_dir}/"config.yaml" ${exp_dir}/"config_stage0.yaml"
  fi
  if [ -f ${exp_dir}/models/"final_model.pt" ]; then
    mv ${exp_dir}/models/"final_model.pt" ${exp_dir}/models/"final_model_stage0.pt"
  fi

  config=conf/whisper_PMFA_stage1.yaml
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  checkpoint=${exp_dir}/models/model_4.pt
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox1_dev/${data_type}.list \
      --train_label ${data}/vox1_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      --model ${model} \
      --checkpoint ${checkpoint}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  model_path=$exp_dir/models/final_model.pt
  echo "Extract embeddings ..."
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 2 --gpus $gpus --data_type raw --data ${data}
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --exp_dir $exp_dir \
    --data ${data} \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox1_dev \
    --top_n $top_n \
    --exp_dir $exp_dir \
    --data ${data} \
    --trials "$trials"
fi
