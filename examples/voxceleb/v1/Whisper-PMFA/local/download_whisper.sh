download_dir=data/whisper_pretrained_model

. tools/parse_options.sh || exit 1

[ ! -d ${download_dir} ] && mkdir -p ${download_dir}

if [ ! -f ${download_dir}/large-v2.pt ]; then
  echo "Downloading large-v2.pt ..."
  wget --no-check-certificate https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt -P ${download_dir}
  md5=$(md5sum ${download_dir}/large-v2.pt | awk '{print $1}')
  [ $md5 != "668764447eeda98eeba5ef7bfcb4cc3d" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1
fi

