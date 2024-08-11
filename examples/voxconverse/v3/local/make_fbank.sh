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

. ./path.sh || exit 1

scp=''
segments=''
store_dir=''
subseg_cmn=true
nj=1

. tools/parse_options.sh

split_dir=$store_dir/split_scp
log_dir=$store_dir/log
mkdir -p $split_dir
mkdir -p $log_dir

# split the scp file to sub_file, and we can use multi-process to extract Fbank feature
file_len=`wc -l $scp | awk '{print $1}'`
subfile_len=$[$file_len / $nj + 1]
prefix='split'
split -l $subfile_len -d -a 3 $scp ${split_dir}/${prefix}_scp_

for suffix in `seq 0 $[$nj-1]`;do
    suffix=`printf '%03d' $suffix`
    scp_subfile=${split_dir}/${prefix}_scp_${suffix}
    write_ark=$store_dir/fbank_${suffix}.ark
    python3 wespeaker/diar/make_fbank.py \
            --scp ${scp_subfile} \
            --segments ${segments} \
            --ark-path ${write_ark} \
            --subseg-cmn ${subseg_cmn} \
            > ${log_dir}/${prefix}.${suffix}.log 2>&1 &
done

wait

cat $store_dir/fbank_*.scp > $store_dir/fbank.scp
echo "Finish make Fbank."
