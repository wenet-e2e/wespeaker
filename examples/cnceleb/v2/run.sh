#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2022 Zhengyang Chen (chenzhengyang117@gmail.com)

. ./path.sh || exit 1;

stage=-1
stop_stage=-1

data=data
data_type="shard"  # shard/raw

config=conf/resnet.yaml
exp_dir=exp/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150
gpus="[0,1]"
num_avg=10
checkpoint=

score_norm_method="asnorm"  # asnorm/snorm
top_n=300
trials="CNC-Eval-Concat.lst CNC-Eval-Avg.lst"

# setup for large margin fine-tuning
do_lm=0
lm_config=conf/resnet_lm.yaml
lm_stage=3
lm_stop_stage=7
lm_exp_dir=exp/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_LM
lm_num_avg=1

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Preparing datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in cnceleb_train eval; do
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

# stage 3 to stage 7, model training and inference
bash local/train_eval.sh --stage ${stage} \
                --stop_stage ${stop_stage} \
                --data ${data} \
                --data_type ${data_type} \
                --config ${config} \
                --exp_dir ${exp_dir} \
                --gpus ${gpus} \
                --num_avg ${num_avg} \
                --checkpoint ${checkpoint} \
                --score_norm_method ${score_norm_method} \
                --top_n ${top_n} \
                --trials ${trials}


# ================== Large margin fine-tuning ==================
# for reference: https://arxiv.org/abs/2206.11699
# It shoule be noted that the large margin fine-tuning
# is optional. It often be used in speaker verification
# challenge to further improve performance. This training
# proces will take longer segment as input and will take
# up more gpu memory.


if [ ${do_lm} -ne 1 ];
    exit 1
fi

# using the pre-trained average model to initialize the LM training
lm_model_dir=${lm_exp_dir}/models
mkdir -p ${lm_model_dir}
cp $exp_dir/models/avg_model.pt ${lm_model_dir}/model_0.pt

# stage 3 to stage 7, model training and inference
bash local/train_eval.sh --stage ${lm_stage} \
                --stop_stage ${lm_stop_stage} \
                --data ${data} \
                --data_type ${data_type} \
                --config ${lm_config} \
                --exp_dir ${lm_exp_dir} \
                --gpus ${gpus} \
                --num_avg ${lm_num_avg} \
                --checkpoint ${lm_model_dir}/model_0.pt \
                --score_norm_method ${score_norm_method} \
                --top_n ${top_n} \
                --trials ${trials}
