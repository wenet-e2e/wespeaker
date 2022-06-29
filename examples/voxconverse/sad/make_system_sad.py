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


import os
import sys
import importlib.util
import functools
import concurrent.futures

import torch

try:
    from utils_vad import get_speech_timestamps, read_audio, init_jit_model
except Exception:
    def module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    utils_vad = module_from_file("utils_vad", "./silero-vad/utils_vad.py")
    from utils_vad import get_speech_timestamps, read_audio, init_jit_model


def read_scp(scp):
    utt_wav_pair = []
    for line in open(scp, 'r'):
        utt, wav = line.strip().split()
        utt_wav_pair.append((utt, wav))
    return utt_wav_pair


def silero_vad(utt_wav_pair, min_duration,
               sampling_rate=16000, threshold=0.4):
    model = init_jit_model('./silero-vad/files/silero_vad.jit')

    utt, wav = utt_wav_pair

    wav = read_audio(wav, sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(
        wav, model, sampling_rate=sampling_rate,
        threshold=threshold)

    vad_result = ""
    for item in speech_timestamps:
        begin = item['start'] / sampling_rate
        end = item['end'] / sampling_rate
        if end - begin >= min_duration:
            vad_result += "{}-{:08d}-{:08d} {} {:.3f} {:.3f}\n".format(
                utt, int(begin * 1000), int(end * 1000), utt, begin, end)
    return vad_result


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    scp = sys.argv[1]
    min_duration = float(sys.argv[2])

    vad = functools.partial(silero_vad, min_duration=min_duration)
    utt_wav_pair_list = read_scp(scp)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(''.join(executor.map(vad, utt_wav_pair_list)), end='')
