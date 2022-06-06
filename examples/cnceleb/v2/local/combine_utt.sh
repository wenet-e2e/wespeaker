#!/bin/bash
# coding:utf-8
# Author: Zhengyang Chen

stage=0
ori_audio_dir=/tmp_data_dir
new_audio_dir=new_data
data_statistics_dir=statistics
audio_path_list=''
min_duration=5
get_dur_nj=40

. utils/parse_options.sh
set -e

statistics_dir=$data_statistics_dir/ori_stat
comb_statistics_dir=$data_statistics_dir/comb_stat
mkdir -p $statistics_dir
mkdir -p $comb_statistics_dir


if [ $stage -le 0 ]; then
    awk -F '[./]' '{print $(NF-2)"/"$(NF-1)" flac -c -d -s "$0" |"}' ${audio_path_list} > ${statistics_dir}/wav.scp
    awk '{print $1}' ${statistics_dir}/wav.scp > ${statistics_dir}/utt
    # here the spk represents speaker and genre
    awk -F- '{print $0,$1}' ${statistics_dir}/utt > ${statistics_dir}/utt2spk
    ./utils/utt2spk_to_spk2utt.pl ${statistics_dir}/utt2spk > ${statistics_dir}/spk2utt
fi

if [ $stage -le 1 ]; then
    #rm ${statistics_dir}/utt2dur
    utils/combine_short_segments.sh --speaker-only true --get_dur_nj ${get_dur_nj} \
        ${statistics_dir} 5 ${comb_statistics_dir}
fi

if [ $stage -le 2 ]; then
    python utils/comb_accd_to_utt2utts.py --ori_data_dir ${ori_audio_dir} --store_data_dir ${new_audio_dir} --utt2utts ${comb_statistics_dir}/utt2utts --num_process 40
fi
