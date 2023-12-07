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


def main(wav_scp, utt2voice_dur, filter_wav_scp, dur_thres=5.0):

    utt2voice_dur_dict = {}
    with open(utt2voice_dur, "r") as f:
        for line in f:
            utt, dur = line.strip().split()
            utt2voice_dur_dict[utt] = float(dur)

    with open(wav_scp, "r") as f, open(filter_wav_scp, "w") as fw:
        for line in f:
            utt = line.strip().split()[0]
            if utt in utt2voice_dur_dict:
                if utt2voice_dur_dict[utt] > dur_thres:
                    fw.write(line)


if __name__ == "__main__":
    fire.Fire(main)
