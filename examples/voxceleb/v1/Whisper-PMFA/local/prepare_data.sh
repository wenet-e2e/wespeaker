#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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
data=data

. tools/parse_options.sh || exit 1

data=`realpath ${data}`
download_dir=${data}/download_data
rawdata_dir=${data}/raw_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Download musan.tar.gz, rirs_noises.zip, vox1_test_wav.zip, and vox1_dev_wav.zip."
  echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

  ./local/download_data.sh --download_dir ${download_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  for archive in musan.tar.gz rirs_noises.zip vox1_test_wav.zip vox1_dev_wav.zip; do
    [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
  done
  [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  if [ ! -d ${rawdata_dir}/musan ]; then
    tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
    unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/voxceleb1 ]; then
    mkdir -p ${rawdata_dir}/voxceleb1/test ${rawdata_dir}/voxceleb1/dev
    unzip ${download_dir}/vox1_test_wav.zip -d ${rawdata_dir}/voxceleb1/test
    unzip ${download_dir}/vox1_dev_wav.zip -d ${rawdata_dir}/voxceleb1/dev
  fi

  echo "Decompress success !!!"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p ${data}/musan ${data}/rirs ${data}/vox1_dev ${data}/vox1_test
  # musan
  find ${rawdata_dir}/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
  # rirs
  find ${rawdata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/rirs/wav.scp
  # vox1 dev
  find ${rawdata_dir}/voxceleb1/dev -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/vox1_dev/wav.scp
  awk '{print $1}' ${data}/vox1_dev/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vox1_dev/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/vox1_dev/utt2spk >${data}/vox1_dev/spk2utt
  # vox1 test
  find ${rawdata_dir}/voxceleb1/test -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/vox1_test/wav.scp
  awk '{print $1}' ${data}/vox1_test/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vox1_test/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/vox1_test/utt2spk >${data}/vox1_test/spk2utt

  if [ ! -d ${data}/vox1_test/trials ]; then
    echo "Download trials for vox1_test ..."
    mkdir -p ${data}/vox1_test/trials
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt -O ${data}/vox1_test/trials/vox1-O.txt
    wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt -O ${data}/vox1_test/trials/vox1-O\(cleaned\).txt
    # transform them into kaldi trial format
    awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/vox1_test/trials/vox1-O\(cleaned\).txt >${data}/vox1_test/trials/vox1_O_cleaned.kaldi
  fi

  echo "Success !!!"
fi
