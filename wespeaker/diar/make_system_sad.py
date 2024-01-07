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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import functools
import concurrent.futures
import argparse
import importlib

import torch
from wespeaker.utils.file_utils import read_scp


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--repo-path',
                        required=True,
                        help='VAD model repo path')
    parser.add_argument('--scp', required=True, help='wav scp')
    parser.add_argument('--min-duration',
                        required=True,
                        type=float,
                        help='min duration')
    args = parser.parse_args()

    return args


def silero_vad(utt_wav_pair,
               repo_path,
               min_duration,
               sampling_rate=16000,
               threshold=0.25):

    def module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    utils_vad = module_from_file("utils_vad",
                                 os.path.join(repo_path, "utils_vad.py"))
    model = utils_vad.init_jit_model(
        os.path.join(repo_path, 'files/silero_vad.jit'))

    utt, wav = utt_wav_pair

    wav = utils_vad.read_audio(wav, sampling_rate=sampling_rate)
    speech_timestamps = utils_vad.get_speech_timestamps(
        wav, model, sampling_rate=sampling_rate, threshold=threshold)

    vad_result = ""
    for item in speech_timestamps:
        begin = item['start'] / sampling_rate
        end = item['end'] / sampling_rate
        if end - begin >= min_duration:
            vad_result += "{}-{:08d}-{:08d} {} {:.3f} {:.3f}\n".format(
                utt, int(begin * 1000), int(end * 1000), utt, begin, end)

    return vad_result


def main():
    args = get_args()

    vad = functools.partial(silero_vad,
                            repo_path=args.repo_path,
                            min_duration=args.min_duration)
    utt_wav_pair_list = read_scp(args.scp)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(''.join(executor.map(vad, utt_wav_pair_list)), end='')


if __name__ == '__main__':
    torch.set_num_threads(1)

    main()
