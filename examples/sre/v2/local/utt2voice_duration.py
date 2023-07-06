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

import fire
from collections import OrderedDict


def main(vad_file, utt2voice_dur):
    utt2voice_dur_dict = OrderedDict()

    with open(vad_file, 'r') as f:
        for line in f.readlines():
            segs = line.strip().split()
            utt, start, end = segs[-3], float(segs[-2]), float(segs[-1])
            if utt not in utt2voice_dur_dict:
                utt2voice_dur_dict[utt] = 0.0
            utt2voice_dur_dict[utt] += end - start

    with open(utt2voice_dur, 'w') as f:
        for utt, duration in utt2voice_dur_dict.items():
            f.write('{} {}\n'.format(utt, duration))


if __name__ == "__main__":
    fire.Fire(main)
