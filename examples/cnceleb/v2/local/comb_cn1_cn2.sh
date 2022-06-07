#!/bin/bash

# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
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

cnceleb1_audio_dir=/data_root/CN-Celeb_flac/data/
cnceleb2_audio_dir=/data_root/CN-Celeb2_flac/data/
min_duration=5
get_dur_nj=60
statistics_dir=statistics
store_data_dir=new_data

. tools/parse_options.sh
set -e

mkdir -p $statistics_dir


# combine the short audios for Cnceleb2
cnceleb2_audio_dir=`realpath $cnceleb2_audio_dir`
# get the paths of all the audio files
find $cnceleb2_audio_dir -name "*.flac" | sort > ${statistics_dir}/cnceleb2_audio_path_list
echo "combine audios for cnceleb2"
bash local/combine_utt.sh --stage 0 \
                    --ori_audio_dir ${cnceleb2_audio_dir} \
                    --new_audio_dir ${store_data_dir}/CN-Celeb2_wav/data \
                    --data_statistics_dir ${statistics_dir}/cnceleb2 \
                    --audio_path_list ${statistics_dir}/cnceleb2_audio_path_list \
                    --min_duration ${min_duration} \
                    --get_dur_nj ${get_dur_nj}


# combine the short audios for Cnceleb1
cnceleb1_audio_dir=`realpath $cnceleb1_audio_dir`
# get the paths of all the audio files
find $cnceleb1_audio_dir -name "*.flac" | awk -F/ '{if($(NF-1)<"id00800"){print $0}}' | sort > ${statistics_dir}/cnceleb1_audio_path_list
echo "combine audios for cnceleb1_dev"
bash local/combine_utt.sh --stage 0 \
                    --ori_audio_dir ${cnceleb1_audio_dir} \
                    --new_audio_dir ${store_data_dir}/CN-Celeb_wav/data \
                    --data_statistics_dir ${statistics_dir}/cnceleb1 \
                    --audio_path_list ${statistics_dir}/cnceleb1_audio_path_list \
                    --min_duration ${min_duration} \
                    --get_dur_nj ${get_dur_nj}

# process the remaining flac data of cnceleb1 to wav data
find $cnceleb1_audio_dir -name "*.flac" | awk -F/ '{if($(NF-1)>="id00800"){print $0}}' | sort > ${statistics_dir}/cnceleb1_eval_audio_path_list
find $cnceleb1_audio_dir/../eval -name "*.flac" | sort > ${statistics_dir}/cnceleb1_enroll_audio_path_list
awk -F '[./]' '{print $(NF-2)"/"$(NF-1)" "$(NF-2)"/"$(NF-1)}' ${statistics_dir}/cnceleb1_eval_audio_path_list > ${statistics_dir}/cnceleb1_eval_utt2utts
awk -F '[./]' '{print $(NF-2)"/"$(NF-1)" "$(NF-2)"/"$(NF-1)}' ${statistics_dir}/cnceleb1_enroll_audio_path_list > ${statistics_dir}/cnceleb1_enroll_utt2utts
echo "combine audios for cnceleb1_eval"
python local/comb_accd_to_utt2utts.py --ori_data_dir ${cnceleb1_audio_dir} \
                                        --store_data_dir ${store_data_dir}/CN-Celeb_wav/data \
                                        --utt2utts ${statistics_dir}/cnceleb1_eval_utt2utts \
                                        --num_process 40
echo "combine audios for cnceleb1_enroll"
python local/comb_accd_to_utt2utts.py --ori_data_dir ${cnceleb1_audio_dir}/../eval \
                                        --store_data_dir ${store_data_dir}/CN-Celeb_wav/eval \
                                        --utt2utts ${statistics_dir}/cnceleb1_enroll_utt2utts \
                                        --num_process 40
