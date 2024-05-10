#!/bin/bash

# Copyright (c) 2022 Hongji Wang    (jijijiang77@gmail.com)
#               2024 Ondrej Odehnal (OndrejOdehnal42@gmail.com)
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

stage=3
stop_stage=4
data=data

. tools/parse_options.sh || exit 1

data=`realpath ${data}`
download_dir=${data}/download_data
rawdata_dir=${data}/raw_data

stage=1

# NOTE: Voxlingua107 already donwloaded
# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#   echo "Download musan.tar.gz, rirs_noises.zip, vox1_test_wav.zip, vox1_dev_wav.zip, and vox2_aac.zip."
#   echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."
# 
#   ./local/download_data.sh --download_dir ${download_dir}
# fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  if [ -d ${download_dir}/VoxLingua107 ]; then
    mkdir -p ${rawdata_dir}/voxlingua107
    echo "$(find ${download_dir}/VoxLingua107/ -name '*.zip')"
    find ${download_dir}/VoxLingua107/ -name '*.zip' -print0 | xargs -0 -I {} -P 11 unzip -n {}  -d ${rawdata_dir}/voxlingua107/
  fi

  echo "Decompress success !!!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p ${data}/voxlingua107

  # # musan
  # find ${rawdata_dir}/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
  # # rirs
  # find ${rawdata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/rirs/wav.scp

  # voxlingua107
  find ${rawdata_dir}/voxlingua107 -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/voxlingua107/wav.scp
  awk '{print $1}' ${data}/voxlingua107/wav.scp | awk -F "/" '{print $0,$2}' >${data}/voxlingua107/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/voxlingua107/utt2spk >${data}/voxlingua107/spk2utt

  echo "Success !!!"
fi
