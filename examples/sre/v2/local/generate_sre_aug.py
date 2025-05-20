# Copyright (c) 2023 Zhengyang Chen
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

import os
import fire


def main(ori_dir, aug_dir, aug_copy_num=2):

    if not os.path.exists(aug_dir):
        os.makedirs(aug_dir)

    read_wav_scp = os.path.join(ori_dir, 'wav.scp')
    aug_wav_scp = os.path.join(aug_dir, 'wav.scp')
    read_utt2spk = os.path.join(ori_dir, 'utt2spk')
    aug_utt2spk = os.path.join(aug_dir, 'utt2spk')
    read_vad = os.path.join(ori_dir, 'vad')
    store_vad = os.path.join(aug_dir, 'vad')

    with open(read_wav_scp, 'r') as f, open(aug_wav_scp, 'w') as wf:
        for line in f:
            line = line.strip().split()
            utt, other_info = line[0], ' '.join(line[1:])
            for i in range(aug_copy_num + 1):
                wf.write(utt + '_copy-' + str(i) + ' ' + other_info + '\n')

    with open(read_utt2spk, 'r') as f, open(aug_utt2spk, 'w') as wf:
        for line in f:
            line = line.strip().split()
            utt, spk = line[0], line[1]
            for i in range(aug_copy_num + 1):
                wf.write(utt + '_copy-' + str(i) + ' ' + spk + '\n')

    with open(read_vad, 'r') as f, open(store_vad, 'w') as wf:
        for line in f:
            line = line.strip().split()
            seg, utt, vad = line[0], line[1], ' '.join(line[2:])
            for i in range(aug_copy_num + 1):
                new_seg = seg + '_copy-' + str(i)
                new_utt = utt + '_copy-' + str(i)
                wf.write(new_seg + ' ' + new_utt + ' ' + vad + '\n')


if __name__ == "__main__":
    fire.Fire(main)
