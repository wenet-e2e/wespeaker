#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)

. ./path.sh || exit 1

stage=2
stop_stage=2

data=data
data_type="shard"  # shard/raw

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # [2] Download voice activity detection model pretrained by Silero Team
    wget -c https://github.com/snakers4/silero-vad/archive/refs/tags/v4.0.zip -O external_tools/silero-vad-v4.0.zip
    unzip -o external_tools/silero-vad-v4.0.zip -d external_tools
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Get vad segmentation for dataset."
    # Set VAD min duration
    min_duration=0.255
    #python3 local/make_system_sad.py \
           #--repo-path external_tools/silero-vad-4.0 \
           #--scp data/data/sre16_major/wav.scp \
           #--min-duration $min_duration > data/sre16_major.vad
    python3 local/make_system_sad.py \
           --repo-path external_tools/silero-vad-4.0 \
           --scp data/data/swbd_sre_morethan_3utts/wav.scp \
           --min-duration $min_duration > data/swbd_sre.vad
    python3 local/make_system_sad.py \
           --repo-path external_tools/silero-vad-4.0 \
           --scp data/data/sre16_eval_test/wav.scp \
           --min-duration $min_duration > data/sre16_eval_test.vad
    python3 local/make_system_sad.py \
           --repo-path external_tools/silero-vad-4.0 \
           --scp data/data/sre16_eval_enroll/wav.scp \
           --min-duration $min_duration > data/sre16_eval_enroll.vad
fi
