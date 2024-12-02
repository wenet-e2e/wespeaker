#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2023 Zhengyang Chen (chenzhengyang117@gmail.com)

. ./path.sh || exit 1

stage=-1
stop_stage=-1

HOST_NODE_ADDR="localhost:29400"
num_nodes=1
job_id=2024

# the sre data should be prepared in kaldi format and stored in the following directory
# only wav.scp, utt2spk and spk2utt files are needed
sre_data_dir=sre_data_dir
data=data
data_type="shard"  # shard/raw
# whether augment the PLDA data
aug_plda_data=0

config=conf/resnet.yaml
exp_dir=exp/ResNet34-TSTP-emb256-fbank40-num_frms200-aug0.6-spFalse-saFalse-Softmax-SGD-epoch150
gpus="[0,1]"
num_avg=10
checkpoint=

trials="trials trials_tgl trials_yue"

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 5  --sre_data_dir ${sre_data_dir} --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Convert train data to ${data_type}..."
  for dset in swbd_sre; do
        python tools/make_shard_list.py --num_utts_per_shard 1000 \
            --num_threads 16 \
            --prefix shards \
            --shuffle \
            --vad_file ${data}/$dset/vad \
            ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
            ${data}/$dset/shards ${data}/$dset/shard.list
  done

  echo "Convert data for PLDA backend training and evaluation to raw format..."
  if [ $aug_plda_data = 0 ];then
      sre_plda_data=sre
  else
      sre_plda_data=sre_aug
  fi
  for dset in ${sre_plda_data} sre16_major sre16_eval_enroll sre16_eval_test; do
        python tools/make_raw_list.py --vad_file ${data}/$dset/vad \
            ${data}/$dset/wav.scp \
            ${data}/$dset/utt2spk ${data}/$dset/raw.list

  done
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/swbd_sre/${data_type}.list \
      --train_label ${data}/swbd_sre/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
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

  echo "Extract embeddings ..."
  local/extract_sre.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 32 --gpus $gpus --data_type raw --data ${data} \
    --reverb_data ${data}/rirs/lmdb \
    --noise_data ${data}/musan/lmdb \
    --aug_plda_data ${aug_plda_data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score using Cosine Distance..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score with adapted PLDA ..."
  local/score_plda.sh \
    --stage 1 --stop-stage 4 \
    --data ${data} \
    --exp_dir $exp_dir \
    --aug_plda_data ${aug_plda_data} \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi
