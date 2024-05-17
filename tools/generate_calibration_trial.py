# Copyright (c) 2024 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2024 Bing Han (hanbing97@sjtu.edu.cn)
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

import fire
import logging
import random
from tqdm import tqdm


def main(utt2dur, trial_path, each_trial_num=10000):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Generate calibration trial ...')
    short_spk2utt = {}
    long_spk2utt = {}

    with open(utt2dur, 'r') as f:
        for line in f.readlines():
            utt, dur = line.strip().split()
            dur = float(dur)
            spk = utt.split('/')[0]

            if 2 < dur < 6:
                if spk not in short_spk2utt:
                    short_spk2utt[spk] = []
                short_spk2utt[spk].append(utt)

            if dur > 6:
                if spk not in long_spk2utt:
                    long_spk2utt[spk] = []
                long_spk2utt[spk].append(utt)

    long_spks = list(long_spk2utt.keys())
    short_spks = list(short_spk2utt.keys())

    for spk in long_spks:
        if spk not in short_spks:
            long_spk2utt.pop(spk, None)
    long_spks = list(long_spk2utt.keys())
    for spk in short_spks:
        if spk not in long_spks:
            short_spk2utt.pop(spk, None)
    short_spks = list(short_spk2utt.keys())

    with open(trial_path, 'w') as f:
        for _ in tqdm(range(each_trial_num // 2)):
            enroll_spk = random.choice(short_spks)
            spk_index = short_spks.index(enroll_spk)
            nontarget_spk = random.choice(short_spks[:spk_index] +
                                          short_spks[spk_index + 1:])

            # short2short
            enroll_utt, test_utt = random.choices(short_spk2utt[enroll_spk],
                                                  k=2)
            f.write("{} {} {}\n".format(enroll_utt, test_utt, 'target'))
            test_utt = random.choice(short_spk2utt[nontarget_spk])
            f.write("{} {} {}\n".format(enroll_utt, test_utt, 'nontarget'))

            # short2long
            enroll_utt = random.choice(short_spk2utt[enroll_spk])
            test_utt = random.choice(long_spk2utt[enroll_spk])
            f.write("{} {} {}\n".format(enroll_utt, test_utt, 'target'))
            test_utt = random.choice(long_spk2utt[nontarget_spk])
            f.write("{} {} {}\n".format(enroll_utt, test_utt, 'nontarget'))

            # long2long
            enroll_utt, test_utt = random.choices(long_spk2utt[enroll_spk],
                                                  k=2)
            f.write("{} {} {}\n".format(enroll_utt, test_utt, 'target'))
            test_utt = random.choice(long_spk2utt[nontarget_spk])
            f.write("{} {} {}\n".format(enroll_utt, test_utt, 'nontarget'))


if __name__ == "__main__":
    fire.Fire(main)
