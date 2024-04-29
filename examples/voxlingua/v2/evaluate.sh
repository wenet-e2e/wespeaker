#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)

# Adapted by: 2024 Ondrej Odehnal OndrejOdehnal42@gmail.com

. ./path.sh || exit 1

stage=1
stop_stage=1

# data=data
# data="/scratch/project/open-28-58/xodehn09/data"
# data="/mnt/proj3/open-27-67/xodehn09/data/16kHz/NAKI/SPLIT"
data="${DATA_DIR}"

export OMP_NUM_THREADS=32
# export LOGLEVEL=DEBUG

data_type="raw"  # shard/raw

# WavLM pre-trained
config=conf/wavlm_base_MHFA_LR.yaml
exp_dir=exp/WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40

gpus="[0]"
checkpoint=

score_norm_method="asnorm"  # asnorm/snorm

# setup for large margin fine-tuning
lm_config=conf/wavlm_base_MHFA_LR_lm.yaml

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare dataset ..."
  ./local/prepare_voxlingua107_dev.sh --stage 2 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert eval data to ${data_type}..."

  for dset in voxlingua107_dev; do
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

  # # Convert all musan data to LMDB
  # python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # # Convert all rirs data to LMDB
  # python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

  # NOTE: Create scores and prints accuracy
  echo "Evalute model ..."
  model_path=$exp_dir/models/avg_model.pt
  tools/evaluate_V2.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 1 --gpus $gpus --data_type ${data_type} --data ${data}

fi
