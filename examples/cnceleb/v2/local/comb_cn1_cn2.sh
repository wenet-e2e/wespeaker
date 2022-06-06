#!/bin/bash
# coding:utf-8
# Author: Zhengyang Chen

cnceleb1_audio_dir=/data_root/CN-Celeb_flac/data/
cnceleb2_audio_dir=/data_root/CN-Celeb2_flac/data/
min_duration=5
get_dur_nj=40
statistics_dir=statistics
store_data_dir=new_data

. utils/parse_options.sh
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
                    --data_statistics_dir statistics/cnceleb2 \
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
                    --data_statistics_dir statistics/cnceleb1 \
                    --audio_path_list ${statistics_dir}/cnceleb1_audio_path_list \
                    --min_duration ${min_duration} \
                    --get_dur_nj ${get_dur_nj}

# process the remaining flac data of cnceleb1 to wav data
find $cnceleb1_audio_dir -name "*.flac" | awk -F/ '{if($(NF-1)>="id00800"){print $0}}' | sort > ${statistics_dir}/cnceleb1_eval_audio_path_list
find $cnceleb1_audio_dir/../eval/enroll -name "*.flac" | sort > ${statistics_dir}/cnceleb1_enroll_audio_path_list
awk -F '[./]' '{print $(NF-2)"/"$(NF-1)" "$(NF-2)"/"$(NF-1)}' ${statistics_dir}/cnceleb1_eval_audio_path_list > ${statistics_dir}/cnceleb1_eval_utt2utts
awk -F '[./]' '{print $(NF-2)"/"$(NF-1)" "$(NF-2)"/"$(NF-1)}' ${statistics_dir}/cnceleb1_enroll_audio_path_list > ${statistics_dir}/cnceleb1_enroll_utt2utts
echo "combine audios for cnceleb1_eval"
python utils/comb_accd_to_utt2utts.py --ori_data_dir ${cnceleb1_audio_dir} \
                                        --store_data_dir ${store_data_dir}/CN-Celeb_wav/data \
                                        --utt2utts ${statistics_dir}/cnceleb1_eval_utt2utts \
                                        --num_process 40
echo "combine audios for cnceleb1_enroll"
python utils/comb_accd_to_utt2utts.py --ori_data_dir ${cnceleb1_audio_dir}/../eval \
                                        --store_data_dir ${store_data_dir}/CN-Celeb_wav/eval \
                                        --utt2utts ${statistics_dir}/cnceleb1_enroll_utt2utts \
                                        --num_process 40
