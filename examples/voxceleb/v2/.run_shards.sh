#!/bin/bash
# coding:utf-8
# Author: Hongji Wang

. ./path.sh

stage=3
stop_stage=3

config=conf/uio.yaml
exp_dir=exp/uio
gpus="[4]"
num_avg=10

. tools/parse_options.sh || exit 1


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  cat data/vox2_dev/utt2spk | awk '{print $2}' | sort | uniq | \
      awk '{print $1, NR - 1}' > data/vox2_dev/spk2id
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python tools/make_shard_list.py --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --seed '777' \
      --shuffle \
      data/vox2_dev/wav.scp data/vox2_dev/utt2spk \
      data/vox2_dev/shards data/vox2_dev/shard.list
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  python wespeaker/bin/train_uio.py --config $config \
      --train_list data/vox2_dev/shard.list \
      --spk2id data/vox2_dev/spk2id \
      --exp_dir ${exp_dir}
fi

