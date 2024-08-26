#!/bin/bash
# Copyright (c) 2022 Xu Xiang
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
sad_type="oracle"

. tools/parse_options.sh

# Prerequisite
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p external_tools

    # [1] Download evaluation toolkit
    wget -c https://github.com/usnistgov/SCTK/archive/refs/tags/v2.4.12.zip -O external_tools/SCTK-v2.4.12.zip
    unzip -o external_tools/SCTK-v2.4.12.zip -d external_tools

    # [2] Download voice activity detection model pretrained by Silero Team
    #wget -c https://github.com/snakers4/silero-vad/archive/refs/tags/v3.1.zip -O external_tools/silero-vad-v3.1.zip
    #unzip -o external_tools/silero-vad-v3.1.zip -d external_tools

    # [3] Download ResNet34 speaker model pretrained by WeSpeaker Team
    mkdir -p pretrained_models

    wget -c https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx -O pretrained_models/voxceleb_resnet34_LM.onnx
fi


# Download VoxConverse dev audios and the corresponding annotations
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p data

    # Download annotations for dev and test sets
    wget -c https://github.com/joonson/voxconverse/archive/refs/heads/master.zip -O data/voxconverse_master.zip
    unzip -o data/voxconverse_master.zip -d data

    # Download dev audios
    mkdir -p data/dev
    wget -c https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_dev_wav.zip -O data/voxconverse_dev_wav.zip
    unzip -o data/voxconverse_dev_wav.zip -d data/dev

    # Create wav.scp for dev audios
    ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/dev/wav.scp

    # Test audios
    # mkdir -p data/test
    # wget -c https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_test_wav.zip -O data/voxconverse_test_wav.zip
    # unzip -o data/voxconverse_test_wav.zip -d data/test
fi


# Voice activity detection
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Set VAD min duration
    min_duration=0.255

    if [[ "x${sad_type}" == "xoracle" ]]; then
        # Oracle SAD: handling overlapping or too short regions in ground truth RTTM
        while read -r utt wav_path; do
            python3 wespeaker/diar/make_oracle_sad.py \
                    --rttm data/voxconverse-master/dev/${utt}.rttm \
                    --min-duration $min_duration
        done < data/dev/wav.scp > data/dev/oracle_sad
    fi

    if [[ "x${sad_type}" == "xsystem" ]]; then
       # System SAD: applying 'silero' VAD
       python3 wespeaker/diar/make_system_sad.py \
               --scp data/dev/wav.scp \
               --min-duration $min_duration > data/dev/system_sad
    fi
fi


# Applying spectral clustering algorithm (need a CUDA enabled GPU)
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    python3 diar/clusterer.py \
            --scp data/dev/wav.scp \
            --segments data/dev/${sad_type}_sad \
            --source pretrained_models/voxceleb_resnet34_LM.onnx \
            --device cuda \
            --output data/dev/${sad_type}_sad_labels
fi


# Convert labels to RTTMs
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    python3 wespeaker/diar/make_rttm.py \
            --labels data/dev/${sad_type}_sad_labels \
            --channel 1 > data/dev/${sad_type}_sad_rttm
fi


# Evaluate the result
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
         -c 0.25 \
         -r <(cat data/voxconverse-master/dev/*.rttm) \
         -s data/dev/${sad_type}_sad_rttm
fi
