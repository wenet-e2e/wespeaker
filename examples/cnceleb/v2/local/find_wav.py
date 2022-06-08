# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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
import argparse
from tqdm import tqdm


def find_all_wav(dirname, extension='.wav'):
    if dirname[-1] != os.sep:
        dirname += os.sep
    for root, _, filenames in tqdm(os.walk(dirname, followlinks=True)):
        wavfiles = [f for f in filenames if f.endswith(extension)]
        if len(wavfiles) > 0:
            for _wav in wavfiles:
                print(os.path.join(root, _wav))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='data',
                        help='dataset_dir')
    parser.add_argument('--extension',
                        type=str,
                        default='wav',
                        help='file extension name')
    args = parser.parse_args()

    find_all_wav(args.data_dir, args.extension)
