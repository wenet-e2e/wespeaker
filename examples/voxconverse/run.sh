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


# Prerequisite [1-4]
# [1] SCTK: evaluation toolkit
git clone https://github.com/usnistgov/SCTK

# [2] voxconverse: ground truth annotation
git clone https://github.com/joonson/voxconverse voxconverse_gt

# [3] silero-vad: pretrained vad model from silero
git clone https://github.com/snakers4/silero-vad

# [4] voxceleb_resnet34.onnx: pretrained speaker model from wespeaker
wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.onnx


# ONNX runtime
pip3 install onnxruntime-gpu=1.11.1 threadpoolctl==3.1.0 scipy==1.8.1 torch torchaudio


# Download and extract dev audio
mkdir -p data/dev
wget -c https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_dev_wav.zip
unzip voxconverse_dev_wav.zip -d data/dev


# Create wav.scp
ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/dev/wav.scp


# Set VAD min duration
min_duration=0.255

# Oracle SAD: handling overlapping or too short regions in ground truth RTTM
while read -r utt wav_path; do
    python3 sad/make_oracle_sad.py voxconverse_gt/dev/${utt}.rttm $min_duration
done < data/dev/wav.scp > data/dev/oracle_sad

# System SAD: applying 'silero' VAD
python3 sad/make_system_sad.py data/dev/wav.scp $min_duration > data/dev/system_sad


# Two 'sad_type' values: "system" or "oracle"
sad_type="oracle"

# Diarization: applying spectral clustering algorithm (need a CUDA enabled GPU)
python3 diar/clusterer.py \
    --scp data/dev/wav.scp \
    --segments data/dev/${sad_type}_sad \
    --source voxceleb_resnet34.onnx \
    --device cuda > data/dev/${sad_type}_sad_labels


# Convert labels to RTTMs
python3 diar/make_rttm.py data/dev/${sad_type}_sad_labels > data/dev/${sad_type}_sad_rttm


# Evaluation
perl SCTK/src/md-eval/md-eval.pl -c 0.25 -r <(cat voxconverse_gt/dev/*.rttm) -s data/dev/${sad_type}_sad_rttm
