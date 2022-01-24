#!/bin/bash
# Copyright 2021 Hongji Wang

. ./path.sh

stage=-1
stop_stage=-1

config=conf/config.yaml
exp_dir=exp/ResNet34_emb512-fbank80-vox2_dev-aug0.6-spFalse-saFalse-ArcMargin-SGD-epoch66
num_avg=10
gpus="[0,1]"

. tools/parse_options.sh || exit 1;

trial_O=
trial_E=
trial_H=

# TODO: local/prepare_data.sh
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preparing datasets..."
    
    for folder in vox2_dev vox1 musan rirs_noises; do
        mkdir -p data/$folder
        echo "Making wav.scp utt2spk .."
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Start training ..."
    num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
        wenet_speaker/bin/train.py --config $config \
                                    --exp_dir ${exp_dir} \
                                    --seed 42 \
                                    --gpus $gpus \
                                    --num_avg ${num_avg}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Do model average ..."
    avg_model=$exp_dir/models/avg_model.pt
    python wenet_speaker/bin/average_model.py \
        --dst_model $avg_model \
        --src_path $exp_dir/models  \
        --num ${num_avg}

    echo "Extract embeddings ..."
    local/extract_vox.sh --exp_dir $exp_dir --model_path $avg_model
fi

# TODO: wenet_speaker/bin/score.py
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

    cat ${exp_dir}/embeddings/vox1/xvector_*.scp > ${exp_dir}/embeddings/vox1/xvector.scp
    cat ${exp_dir}/embeddings/vox2_dev/xvector_*.scp > ${exp_dir}/embeddings/vox2_dev/xvector.scp

    echo "Python scoring ..."
    python wenet_speaker/bin/score.py \
        --exp_dir ${exp_dir} \
        --eval_scp_path ${exp_dir}/embeddings/vox1/xvector.scp \
        --cal_mean_dir ${exp_dir}/embeddings/vox2_dev \
        --cal_mean True \
        --p_target 0.01 \
        --c_miss 1 \
        --c_fa 1 \
        ${trial_O} ${trial_E} ${trial_H}
fi

