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
import functools
import concurrent.futures
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--repo-or-dir', required=True,
                        help='VAD model repo/dir')
    parser.add_argument('--model', required=True,
                        help="entrypoint defined in the repo/dir's hubconf.py")
    parser.add_argument('--scp', required=True, help='wav scp')
    parser.add_argument('--min-duration', required=True,
                        type=float, help='min duration')
    args = parser.parse_args()

    return args


def read_scp(scp):
    utt_wav_pair = []
    for line in open(scp, 'r'):
        utt, wav = line.strip().split()
        utt_wav_pair.append((utt, wav))

    return utt_wav_pair


def silero_vad(utt_wav_pair, repo_or_dir, model, min_duration,
               sampling_rate=16000, threshold=0.36):
    model, utils = torch.hub.load(repo_or_dir=repo_or_dir,
                                  model=model,
                                  force_reload=False,
                                  onnx=False,
                                  skip_validation=True,
                                  source='local')
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

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


def main():
    args = get_args()

    vad = functools.partial(silero_vad,
                            repo_or_dir=args.repo_or_dir,
                            model=args.model,
                            min_duration=args.min_duration)
    utt_wav_pair_list = read_scp(args.scp)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(''.join(executor.map(vad, utt_wav_pair_list)), end='')


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    main()
