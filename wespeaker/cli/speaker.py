# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys

import librosa
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm

from wespeaker.cli.hub import Hub
from wespeaker.cli.fbank import logfbank


class Speaker:
    def __init__(self, model_path: str, resample_rate: int = 16000):
        self.session = ort.InferenceSession(model_path)
        self.resample_rate = resample_rate
        self.table = {}

    def extract_embedding(self, audio_path: str):
        pcm, sample_rate = librosa.load(audio_path, sr=self.resample_rate)
        pcm = pcm * (1 << 15)
        # NOTE: produce the same results as with torchaudio.compliance.kaldi
        feats = logfbank(
            pcm,
            sample_rate,
            nfilt=80,
            lowfreq=20,
            winlen=0.025,  # 25 ms
            winstep=0.01,  # 10 ms
            dither=0,
            wintype='hamming')
        feats = feats - np.mean(feats, axis=0)  # CMN
        feats = np.expand_dims(feats, axis=0).astype(np.float32)
        outputs = self.session.run(None, {"feats": feats})
        embedding = outputs[0][0]
        return embedding

    def compute_similarity(self, audio_path1: str, audio_path2) -> float:
        e1 = self.extract_embedding(audio_path1)
        e2 = self.extract_embedding(audio_path2)
        return self.cosine_distance(e1, e2)

    def cosine_distance(self, e1, e2):
        return np.dot(e1, e2) / (norm(e1) * norm(e2))

    def register(self, name: str, audio_path: str):
        if name in self.table:
            print('Speaker {} already registered, ignore'.format(name))
        else:
            self.table[name] = self.extract_embedding(audio_path)

    def recognize(self, audio_path: str):
        q = self.extract_embedding(audio_path)
        best_score = 0.0
        best_name = ''
        for name, e in self.table.items():
            score = self.cosine_distance(q, e)
            if best_score < score:
                best_score = score
                best_name = name
        result = {}
        result['name'] = name
        result['confidence'] = best_score
        return result


def load_model(language: str, resample_rate: int) -> Speaker:
    model_path = Hub.get_model(language)
    return Speaker(model_path, resample_rate)


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t',
                        '--task',
                        choices=[
                            'embedding',
                            'similarity',
                            'diarization',
                        ],
                        default='embedding',
                        help='task type')
    parser.add_argument('-l',
                        '--language',
                        choices=[
                            'chinese',
                            'english',
                        ],
                        default='chinese',
                        help='language type')
    parser.add_argument('--audio_file', help='audio file')
    parser.add_argument('--audio_file2',
                        help='audio file2, for similarity task')
    parser.add_argument('--resample_rate',
                        type=int,
                        default=16000,
                        help='resampling rate')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = load_model(args.language, args.resample_rate)
    if args.task == 'embedding':
        print(model.extract_embedding(args.audio_file))
    elif args.task == 'similarity':
        print(model.compute_similarity(args.audio_file, args.audio_file2))
    elif args.task == 'diarization':
        # TODO(Chengdong Liang): Add diarization surport
        pass
    else:
        print('Unsupported task {}'.format(args.task))
        sys.exit(-1)


if __name__ == '__main__':
    main()
