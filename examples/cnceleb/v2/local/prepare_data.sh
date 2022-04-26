#!/bin/bash
# coding:utf-8
# Author: Hongji Wang, Chengdong Liang

stage=-1
stop_stage=-1

. tools/parse_options.sh || exit 1

download_dir=data/download_data
rawdata_dir=data/raw_data
# download_dir=/home/yaojiadi/asv/cnceleb
# rawdata_dir=/home/yaojiadi/asv/cnceleb/raw_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Download musan.tar.gz, rirs_noises.zip, cn-celeb_v2.tar.gz and cn-celeb2_v2.tar.gz."
  echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

  ./local/download_data.sh --download_dir ${download_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  for archive in musan.tar.gz rirs_noises.zip cn-celeb_v2.tar.gz cn-celeb2_v2.tar.gz; do
    [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
  done
  [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  if [ ! -d ${rawdata_dir}/musan ]; then
    tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
    unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/CN-Celeb_flac ]; then
    tar -xzvf ${download_dir}/cn-celeb_v2.tar.gz -C ${raw_data}
  fi

  if [ ! -d ${rawdata_dir}/CN-Celeb2_flac ]; then
    tar -xzvf ${download_dir}/cn-celeb2_v2.tar.gz -C ${raw_data}
  fi

  echo "Decompress success !!!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "convert flac to wav ..."
  python local/flac2wav.py \
    --dataset_dir ${rawdata_dir}/CN-Celeb_flac \
    --nj 16

  python local/flac2wav.py \
    --dataset_dir ${rawdata_dir}/CN-Celeb2_flac \
    --nj 16
  echo "convert success"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p data/musan data/rirs data/cnceleb_train data/eval
  # musan
  find $(pwd)/${rawdata_dir}/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >data/musan/wav.scp || exit 1;
  # rirs
  find $(pwd)/${rawdata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >data/rirs/wav.scp || exit 1;

  echo "Prepare train data including CN-Celeb_wav/dev and CN-Celeb2_wav ..."
  mkdir -p data/train
  for spk in `cat ${rawdata_dir}/CN-Celeb_flac/dev/dev.lst`; do
    if [ ! -d data/train/$spk ]; then
      ln -s $(pwd)/${rawdata_dir}/CN-Celeb_wav/data/${spk} data/train/${spk}
    fi
  done

  for spk in `cat ${rawdata_dir}/CN-Celeb2_flac/spk.lst`; do
    if [ ! -d data/train/$spk ]; then
      ln -s $(pwd)/${rawdata_dir}/CN-Celeb2_wav/data/${spk} data/train/${spk}
    fi
  done
  echo "Prepare data for training ..."
  python local/find_wav.py --data_dir data/train --extension wav |\
     awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >data/cnceleb_train/wav.scp
  awk '{print $1}' data/cnceleb_train/wav.scp | awk -F "/" '{print $0,$1}' >data/cnceleb_train/utt2spk
  ./tools/utt2spk_to_spk2utt.pl data/cnceleb_train/utt2spk >data/cnceleb_train/spk2utt

  echo "Prepare data for testing ..."
  find $(pwd)/${rawdata_dir}/CN-Celeb_wav/eval -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >data/eval/wav.scp
  awk '{print $1}' data/eval/wav.scp | awk -F "/" '{print $0,$1}' >data/eval/utt2spk
  ./tools/utt2spk_to_spk2utt.pl data/eval/utt2spk >data/eval/spk2utt

  echo "Prepare evalution trials ..."
  mkdir -p data/eval/trials
  python local/format_trials_cnceleb.py \
    --cnceleb_root $(pwd)/${rawdata_dir}/CN-Celeb_flac \
    --dst_trl_path data/eval/trials/CNC-Eval-Core.lst

  echo "Success !!! Now data preparation is done !!!"
fi
