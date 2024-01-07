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
pretrained_model=''
device=cuda
store_dir=''
subseg_cmn=true
nj=1

batch_size=96
frame_shift=10
window_secs=1.5
period_secs=0.75

. tools/parse_options.sh

split_dir=$store_dir/split_scp
log_dir=$store_dir/log
mkdir -p $split_dir
mkdir -p $log_dir

# split the scp file to sub_file, and we can use multi-process to extract embeddings
file_len=`wc -l $scp | awk '{print $1}'`
subfile_len=$[$file_len / $nj + 1]
prefix='split'
split -l $subfile_len -d -a 3 $scp ${split_dir}/${prefix}_scp_

for suffix in `seq 0 $[$nj-1]`;do
    suffix=`printf '%03d' $suffix`
    scp_subfile=${split_dir}/${prefix}_scp_${suffix}
    write_ark=$store_dir/emb_${suffix}.ark
    python3 wespeaker/diar/extract_emb.py \
            --scp ${scp_subfile} \
            --ark-path ${write_ark} \
            --source ${pretrained_model} \
            --device ${device} \
            --batch-size ${batch_size} \
            --frame-shift ${frame_shift} \
            --window-secs ${window_secs} \
            --period-secs ${period_secs} \
            --subseg-cmn ${subseg_cmn} \
            > ${log_dir}/${prefix}.${suffix}.log 2>&1 &
done

wait

cat $store_dir/emb_*.scp > $store_dir/emb.scp
echo "Finish extract embedding."
