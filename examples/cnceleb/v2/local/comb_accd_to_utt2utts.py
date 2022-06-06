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

import os
import numpy as np
import fire
import soundfile as sf
import pypeln as pl
from tqdm import tqdm


def store_comb_data_sub_process(tuple_data):
    '''
    tuple: (ori_data_dir, store_data_dir, line)
    line: spk_id/utt-comb spk_id/utt1 spk_id/utt2 ...
    '''
    ori_data_dir, store_data_dir, line = tuple_data

    segs = line.strip().split()

    spk_id = segs[0].split('/')[0]
    store_dir = os.path.join(store_data_dir, spk_id)
    os.makedirs(store_dir, exist_ok=True)

    store_path = os.path.join(store_data_dir, segs[0] + '.wav')

    data_list = []
    for utt_name in segs[1:]:
        utt_path = os.path.join(ori_data_dir, utt_name + '.flac')
        data, sr = sf.read(utt_path)
        data_list.append(data)

    data = np.concatenate(data_list)

    sf.write(store_path, data, sr)

    return 0


def store_comb_data(ori_data_dir, store_data_dir, utt2utts, num_process=10):

    with open(utt2utts, 'r') as f:
        lines = f.readlines()
        lines_num = len(lines)

        ori_data_dir_list = [ori_data_dir] * lines_num
        store_data_dir_list = [store_data_dir] * lines_num

        t_bar = tqdm(ncols=100, total=lines_num)
        for _ in pl.process.map(store_comb_data_sub_process,
                                zip(ori_data_dir_list, store_data_dir_list,
                                    lines),
                                workers=num_process,
                                maxsize=num_process + 1):
            t_bar.update()

        t_bar.close()


if __name__ == "__main__":
    fire.Fire(store_comb_data)
