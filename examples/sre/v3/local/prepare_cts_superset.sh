#!/bin/bash

set -o pipefail

export LC_ALL=C


data_cts=data/cts/
cts_superset_dir=""
wav_dir=wav/cts/

. tools/parse_options.sh || exit 1

echo $cts_superset_dir

if [ ! -f $cts_superset_dir/docs/cts_superset_segment_key.tsv ];then
    echo "ERROR: $cts_superset_dir/docs/cts_superset_segment_key.tsv does not exist."
    exit 1
fi

mkdir -p $data_cts


echo -n "" > ${data_cts}/wav.scp
for x in $(tail -n +2 $cts_superset_dir/docs/cts_superset_segment_key.tsv | cut -f 1 | sed "s:\.sph::" );do
    echo "$x ffmpeg -nostdin -i ${cts_superset_dir}/data/${x}.sph -ar 8000 -f wav pipe:1 |" >> $data_cts/wav.scp
done


tail -n +2 $cts_superset_dir/docs/cts_superset_segment_key.tsv | cut -f 1,3 --output-delimiter=" " | sed "s:\.sph::" | sort > ${data_cts}/utt2spk

tools/utt2spk_to_spk2utt.pl ${data_cts}/utt2spk > ${data_cts}/spk2utt
