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


# Prerequisite
# SCTK: evaluation toolkit
# voxconverse: ground truth annotation
# silero-vad: system vad
git clone https://github.com/usnistgov/SCTK
git clone https://github.com/joonson/voxconverse voxconverse_gt
git clone https://github.com/snakers4/silero-vad

# github size limit
cat avg_model.onnx.part1 avg_model.onnx.part2 > avg_model.onnx 

# Download and extract dev audio
mkdir -p data/dev
wget https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_dev_wav.zip
unzip voxconverse_dev_wav.zip -d data/dev

# Create wav.scp
ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 0, length($NF)-4), $0}' > data/dev/wav.scp

# Set VAD min duration
min_duration=0.255

# Oracle SAD: handling overlapping or too short regions in ground truth RTTM
while read -r utt wav_path; do
    python3 sad/make_oracle_sad.py voxconverse_gt/dev/${utt}.rttm $min_duration
done < data/dev/wav.scp > data/dev/oracle_sad

# System SAD: applying 'silero' VAD
system_sad_dir=data/dev/system_sad_results
rm -rf ${system_sad_dir} && mkdir ${system_sad_dir}

# Create VAD tasks
while read -r utt wav_path; do
    echo "\$(python3 sad/make_system_sad.py <(echo ${utt} ${wav_path}) $min_duration > ${system_sad_dir}/${utt}.system_sad)"
done < data/dev/wav.scp > data/dev/system_sad_tasks.sh

# Run VAD tasks
num_procs=72
cat data/dev/system_sad_tasks.sh | xargs -P ${num_procs} -i bash -c "{}"
cat ${system_sad_dir}/*.system_sad > data/dev/system_sad

# Two 'sad_type' values: "system" or "oracle"
sad_type="system"
label_dir=data/dev/labels
rm -rf ${label_dir} && mkdir ${label_dir}
while read -r utt wav_path; do
    echo "\$(python3 diar/clusterer.py --onnx-model avg_model.onnx --wav-scp <(echo ${utt} ${wav_path}) --segments data/dev/${sad_type}_sad > ${label_dir}/${utt}.labels)"
done < data/dev/wav.scp > data/dev/embedding_extraction_tasks.sh

# RUN embedding extraction tasks
num_procs=72
cat data/dev/embedding_extraction_tasks.sh | xargs -P ${num_procs} -i bash -c "{}"

# Convert labels to RTTMs
rttm_dir=data/dev/rttms
rm -rf ${rttm_dir} && mkdir ${rttm_dir}
while read -r utt wav_path; do
    python3 diar/make_rttm.py ${label_dir}/${utt}.labels > ${rttm_dir}/${utt}.rttm
done < data/dev/wav.scp

# Evaluation
perl SCTK/src/md-eval/md-eval.pl -c 0.25 -r <(cat voxconverse_gt/dev/*.rttm) -s <(cat ${rttm_dir}/*.rttm)
