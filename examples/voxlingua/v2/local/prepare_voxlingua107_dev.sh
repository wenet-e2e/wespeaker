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
dataset_dir=voxlingua107_dev
rawdata_dir=${data}/${dataset_dir}/raw_data

stage=1

# NOTE: Voxlingua107 dev already donwloaded

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p ${data}/${dataset_dir}

  # # musan
  # find ${rawdata_dir}/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
  # # rirs
  # find ${rawdata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/rirs/wav.scp

  # voxlingua107
  find ${rawdata_dir} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/${dataset_dir}/wav.scp
  awk '{print $1}' ${data}/${dataset_dir}/wav.scp | awk -F "/" '{print $0,$2}' >${data}/${dataset_dir}/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/${dataset_dir}/utt2spk >${data}/${dataset_dir}/spk2utt

  echo "Success !!!"
fi

