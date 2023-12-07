# Copyright (c) 2022 Xu Xiang
#               2023 Zhengyang Chen
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
import io
import functools
import concurrent.futures
import argparse
import importlib
import torchaudio
import subprocess

import torch


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


@functools.lru_cache(maxsize=1)
def load_wav(wav_rxfilename, ):
    """ This function reads audio file and return data in pytorch tensor.
        "lru_cache" holds recently loaded audio so that can be called
        many times on the same audio file.
        OPTIMIZE: controls lru_cache size for random access,
        considering memory size
    """
    if wav_rxfilename.endswith('|'):
        # input piped command
        p = subprocess.Popen(wav_rxfilename[:-1],
                             shell=True,
                             stdout=subprocess.PIPE)
        data, samplerate = torchaudio.load(io.BytesIO(p.stdout.read()))
    elif wav_rxfilename == '-':
        # stdin
        data, samplerate = torchaudio.load(sys.stdin)
    else:
        # normal wav file
        data, samplerate = torchaudio.load(wav_rxfilename)
    return data.squeeze(0), samplerate


def read_scp(scp):
    utt_wav_pair = []
    for line in open(scp, 'r'):
        segs = line.strip().split()
        if len(segs) > 2:
            utt, wav = segs[0], ' '.join(segs[1:])
        else:
            utt, wav = segs[0], segs[1]
        utt_wav_pair.append((utt, wav))

    return utt_wav_pair


def silero_vad(utt_wav_pair,
               repo_path,
               min_duration,
               sampling_rate=8000,
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

    wav, sr = load_wav(wav)
    assert sr == sampling_rate
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
