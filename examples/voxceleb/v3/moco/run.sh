#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#           2023 Bing Han (hanbing97@sjtu.edu.cn)

. ./path.sh || exit 1

stage=3
stop_stage=6
use_slurm=0

data=data
data_type="shard"  # shard/raw

config=conf/ecapa_tdnn_moco.yaml # simclr for conf/ecapa_tdnn_simclr.yaml
exp_dir=exp/ECAPA_TDNN_GLOB_c512-ASTP-emb192-fbank80-num_frms200-aug1.0-spTrue-saFalse-MoCo-SGD-epoch150
gpus="[0,1,2,3]"
num_avg=10
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

. tools/parse_options.sh || exit 1

if [ $use_slurm -eq 1 ];then
    python_cmd="srun /mnt/lustre/sjtu/home/czy97/.conda/envs/pytorch1_12/bin/python"
else
    num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
    python_cmd="torchrun --master_addr=localhost --master_port=16888 --nnodes=1 --nproc_per_node=$num_gpus"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox2_dev vox1; do
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
  echo "Start training ..."
  $python_cmd local/train_contrastive.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --wav_scp ${data}/vox2_dev/wav.scp \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python local/average_contrastive_model.py \
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
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi