#!/bin/bash

stage=1
stop_stage=4
data=data
proj=NAKI_SPLIT

. tools/parse_options.sh || exit 1

# rawdata_dir="/mnt/proj3/open-27-67/xodehn09/data/16kHz/NAKI/SPLIT"
# rawdata_dir="/pfs/lustrep1/scratch/project_465000792/xodehn09/data/NAKI_SPLIT/"
data="/pfs/lustrep1/scratch/project_465000792/xodehn09/data"
proj="NAKI_filtered/test/NAKI_split"
rawdata_dir="$data/$proj"

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p ${data}/$proj

  # NOTE: Already processed
  # # musan
  # find ${rawdata_dir}/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
  # # rirs
  # find ${rawdata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/rirs/wav.scp

  # naki
  find ${rawdata_dir} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/$proj/wav.scp
  awk '{print $1}' ${data}/$proj/wav.scp | awk -F "/" '{print $0,$2}' > ${data}/$proj/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/$proj/utt2spk > ${data}/$proj/spk2utt

  echo "Success !!!"
fi

python tools/make_raw_list.py ${data}/$proj/wav.scp ${data}/$proj/utt2spk ${data}/$proj/raw.list

