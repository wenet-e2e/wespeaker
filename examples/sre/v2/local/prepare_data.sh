#!/bin/bash

# Copyright (c) 2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

stage=-1
stop_stage=-1
sre_data_dir=
data=data

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p external_tools
    # Download voice activity detection model pretrained by Silero Team
    wget -c https://github.com/snakers4/silero-vad/archive/refs/tags/v4.0.zip -O external_tools/silero-vad-v4.0.zip
    unzip -o external_tools/silero-vad-v4.0.zip -d external_tools
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # The meta data for SRE16 should be pre-prepared using Kaldi recipe:
    # https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16/v2
    for dset in swbd_sre sre sre16_major sre16_eval_enroll sre16_eval_test; do
        mkdir -p ${data}/${dset}
        cp ${sre_data_dir}/${dset}/wav.scp ${data}/${dset}/wav.scp
        [ -f ${sre_data_dir}/${dset}/utt2spk ] && cp ${sre_data_dir}/${dset}/utt2spk ${data}/${dset}/utt2spk
        [ -f ${sre_data_dir}/${dset}/spk2utt ] && cp ${sre_data_dir}/${dset}/spk2utt ${data}/${dset}/spk2utt
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Get vad segmentation for dataset."
    # Set VAD min duration
    min_duration=0.255
    for dset in swbd_sre sre16_major sre16_eval_enroll sre16_eval_test; do
        python3 local/make_system_sad.py \
               --repo-path external_tools/silero-vad-4.0 \
               --scp ${data}/${dset}/wav.scp \
               --min-duration $min_duration > ${data}/${dset}/vad
    done
    tools/filter_scp.pl -f 2 ${data}/sre/wav.scp ${data}/swbd_sre/vad > ${data}/sre/vad

    # For PLDA training, it is better to augment the training data
    python3 local/generate_sre_aug.py --ori_dir ${data}/sre \
                                    --aug_dir ${data}/sre_aug \
                                    --aug_copy_num 2
    tools/utt2spk_to_spk2utt.pl ${data}/sre_aug/utt2spk > ${data}/sre_aug/spk2utt

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    for dset in swbd_sre; do
        python3 local/utt2voice_duration.py \
            --vad_file ${data}/${dset}/vad \
            --utt2voice_dur ${data}/${dset}/utt2voice_dur
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Following the Kaldi recipe: https://github.com/kaldi-asr/kaldi/blob/71f38e62cad01c3078555bfe78d0f3a527422d75/egs/sre16/v2/run.sh#L189
    # We filter out the utterances with duration less than 5s
    for dset in swbd_sre; do
        python3 local/filter_utt_accd_dur.py \
            --wav_scp ${data}/${dset}/wav.scp \
            --utt2voice_dur ${data}/${dset}/utt2voice_dur \
            --filter_wav_scp ${data}/${dset}/filter_wav.scp \
            --dur_thres 5.0
        mv ${data}/${dset}/wav.scp ${data}/${dset}/wav.scp.bak
        mv ${data}/${dset}/filter_wav.scp ${data}/${dset}/wav.scp
    done

    # Similarly, following the Kaldi recipe,
    # we throw out speakers with fewer than 3 utterances.
    for dset in swbd_sre; do
        tools/fix_data_dir.sh ${data}/${dset}
        cp ${data}/${dset}/spk2utt ${data}/${dset}/spk2utt.bak
        awk '{if(NF>2){print $0}}' ${data}/${dset}/spk2utt.bak > ${data}/${dset}/spk2utt
        tools/spk2utt_to_utt2spk.pl ${data}/${dset}/spk2utt > ${data}/${dset}/utt2spk
        tools/fix_data_dir.sh ${data}/${dset}
    done
fi
