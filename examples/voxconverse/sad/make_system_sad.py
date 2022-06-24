# Copyright (c) 2022 Xu Xiang
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

import sys
import torch

def silero_vad(wav_scp, min_duration):
    USE_ONNX = True
    SAMPLING_RATE = 16000
    repo_or_dir = "./silero-vad"
    model, utils = torch.hub.load(repo_or_dir=repo_or_dir,
                                  model='silero_vad',
                                  force_reload=False,
                                  skip_validation=True,
                                  source='local',
                                  onnx=USE_ONNX)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    for line in open(wav_scp, 'r'):
        utt, wav_path = line.strip().split()
        wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(
            wav, model, sampling_rate=SAMPLING_RATE)
        for item in speech_timestamps:
            begin = item['start'] / SAMPLING_RATE
            end = item['end'] / SAMPLING_RATE
            if end - begin >= min_duration:
                print("{}-{:08d}-{:08d} {} {:.3f} {:.3f}".format(
                    utt, int(begin * 1000), int(end * 1000), utt, begin, end))

if __name__ == '__main__':
    torch.set_num_threads(1)
    silero_vad(sys.argv[1], float(sys.argv[2]))
