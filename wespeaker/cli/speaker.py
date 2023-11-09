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
from silero_vad import vad

from wespeaker.cli.hub import Hub
from wespeaker.cli.fbank import logfbank


class Speaker:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.vad_model = vad.OnnxWrapper()
        self.table = {}
        self.resample_rate = 16000
        self.apply_vad = True

    def set_resample_rate(self, resample_rate: int):
        self.resample_rate = resample_rate

    def set_vad(self, apply_vad: bool):
        self.apply_vad = apply_vad

    def extract_embedding(self, audio_path: str):
        pcm, sample_rate = librosa.load(audio_path, sr=self.resample_rate)
        pcm = pcm * (1 << 15)
        if self.apply_vad:
            # TODO(Binbin Zhang): Refine the segments logic, here we just
            # suppose there is only silence at the start/end of the speech
            segments = vad.get_speech_timestamps(self.vad_model,
                                                 audio_path,
                                                 return_seconds=True)
            if len(segments) > 0:  # remove head and tail silence
                start = int(segments[0]['start'] * sample_rate)
                end = int(segments[-1]['end'] * sample_rate)
                pcm = pcm[start:end]
            else:  # all silence, nospeech
                return None

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

    def compute_similarity(self, audio_path1: str, audio_path2: str) -> float:
        e1 = self.extract_embedding(audio_path1)
        e2 = self.extract_embedding(audio_path2)
        if e1 is None or e2 is None:
            return 0.0
        else:
            return self.cosine_similarity(e1, e2)

    def cosine_similarity(self, e1, e2):
        cosine_score = np.dot(e1, e2) / (norm(e1) * norm(e2))
        return (cosine_score + 1.0) / 2  # normalize: [-1, 1] => [0, 1]

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
            score = self.cosine_similarity(q, e)
            if best_score > score:
                best_score = score
                best_name = name
        result = {}
        result['name'] = name
        result['confidence'] = best_score
        return result

    def diarize(self, audio_path: str):
        #  TODO
        pcm, sample_rate = librosa.load(audio_path, sr=self.resample_rate)
        return [[0.0, len(pcm)/sample_rate, 0]]


def load_model(language: str) -> Speaker:
    model_path = Hub.get_model(language)
    return Speaker(model_path)


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
                        help='audio file2, specifically for similarity task')
    parser.add_argument('--resample_rate',
                        type=int,
                        default=16000,
                        help='resampling rate')
    parser.add_argument('--vad',
                        action='store_true',
                        help='whether to do VAD or not')
    parser.add_argument('--output_file',
                        help='output file to save speaker embedding or diarization result')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = load_model(args.language)
    model.set_resample_rate(args.resample_rate)
    model.set_vad(args.vad)
    if args.task == 'embedding':
        embedding = model.extract_embedding(args.audio_file)
        if embedding is not None:
            np.savetxt(args.output_file, embedding)
            print('Succeed, see {}'.format(args.output_file))
        else:
            print('Fails to extract embedding')
    elif args.task == 'similarity':
        print(model.compute_similarity(args.audio_file, args.audio_file2))
    elif args.task == 'diarization':
        # TODO(Chengdong Liang): Add diarization surport
        diar_result = model.diarize(args.audio_file)
        with open(args.output_file, "w") as fout:
            for (start, end, spkid) in diar_result:
                fout.write("{:.3f}\t{:.3f}\t{:d}\n".format(start, end, spkid))
        print('Succeed, see {}'.format(args.output_file))
    else:
        print('Unsupported task {}'.format(args.task))
        sys.exit(-1)


if __name__ == '__main__':
    main()
