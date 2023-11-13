# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#                    Shuai Wang (wsstriving@gmail.com)
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
import os
import sys

import numpy as np
from silero_vad import vad
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
import kaldiio
from tqdm import tqdm

from wespeaker.cli.hub import Hub
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


class Speaker:
    def __init__(self, model_dir: str):
        config_path = os.path.join(model_dir, 'config.yaml')
        model_path = os.path.join(model_dir, 'avg_model.pt')
        with open(config_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        self.model = get_speaker_model(
            configs['model'])(**configs['model_args'])
        load_checkpoint(self.model, model_path)
        self.vad_model = vad.OnnxWrapper()
        self.table = {}
        self.resample_rate = 16000
        self.apply_vad = False
        self.device = torch.device('cpu')

    def set_resample_rate(self, resample_rate: int):
        self.resample_rate = resample_rate

    def set_vad(self, apply_vad: bool):
        self.apply_vad = apply_vad

    def set_gpu(self, device_id: int):
        if device_id >= 0:
            device = 'cuda:{}'.format(device_id)
        else:
            device = 'cpu'
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def extract_embedding(self, audio_path: str):
        pcm, sample_rate = torchaudio.load(audio_path, normalize=False)
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
        pcm = pcm.to(torch.float)
        if sample_rate != self.resample_rate:
            pcm = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate)(pcm)
        feats = kaldi.fbank(pcm,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            energy_floor=0.0,
                            sample_frequency=self.resample_rate)
        feats = feats - torch.mean(feats, 0)  # CMN
        feats = feats.unsqueeze(0)
        feats = feats.to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, outputs = self.model(feats)
        embedding = outputs[0].to(torch.device('cpu'))
        return embedding

    def extract_embedding_list(self, scp_path: str):
        names = []
        embeddings = []
        with open(scp_path, 'r') as read_scp:
            for line in tqdm(read_scp):
                name, wav_path = line.strip().split()
                names.append(name)
                embedding = self.extract_embedding(wav_path)
                embeddings.append(embedding.detach().numpy())
        return names, embeddings

    def compute_similarity(self, audio_path1: str, audio_path2: str) -> float:
        e1 = self.extract_embedding(audio_path1)
        e2 = self.extract_embedding(audio_path2)
        if e1 is None or e2 is None:
            return 0.0
        else:
            return self.cosine_similarity(e1, e2)

    def cosine_similarity(self, e1, e2):
        cosine_score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
        cosine_score = cosine_score.item()
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
        return [[0.0, len(pcm) / sample_rate, 0]]


def load_model(language: str) -> Speaker:
    model_path = Hub.get_model(language)
    return Speaker(model_path)


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t',
                        '--task',
                        choices=[
                            'embedding',
                            'embedding_kaldi',
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
    parser.add_argument('-g',
                        '--gpu',
                        type=int,
                        default=-1,
                        help='which gpu to use (number <0 means using cpu)')
    parser.add_argument('--audio_file', help='audio file')
    parser.add_argument('--audio_file2',
                        help='audio file2, specifically for similarity task')
    parser.add_argument('--wav_scp',
                        help='path to wav.scp, for extract and saving '
                             'kaldi-stype embeddings')
    parser.add_argument('--resample_rate',
                        type=int,
                        default=16000,
                        help='resampling rate')
    parser.add_argument('--vad',
                        action='store_true',
                        help='whether to do VAD or not')
    parser.add_argument('--output_file',
                        help='output file to save speaker embedding')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = load_model(args.language)
    model.set_resample_rate(args.resample_rate)
    model.set_vad(args.vad)
    model.set_gpu(args.gpu)
    if args.task == 'embedding':
        embedding = model.extract_embedding(args.audio_file)
        if embedding is not None:
            np.savetxt(args.output_file, embedding.detach().numpy())
            print('Succeed, see {}'.format(args.output_file))
        else:
            print('Fails to extract embedding')
    elif args.task == 'embedding_kaldi':
        names, embeddings = model.extract_embedding_list(args.wav_scp)
        embed_ark = args.output_file + ".ark"
        embed_scp = args.output_file + ".scp"
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," + embed_scp) as writer:
            for name, embedding in zip(names, embeddings):
                writer(name, embedding)
    elif args.task == 'similarity':
        print(model.compute_similarity(args.audio_file, args.audio_file2))
    elif args.task == 'diarization':
        # TODO(Chengdong Liang): Add diarization surport
        diar_result = model.diarize(args.audio_file)
        for (start, end, spkid) in diar_result:
            print("{:.3f}\t{:.3f}\t{:d}".format(start, end, spkid))
    else:
        print('Unsupported task {}'.format(args.task))
        sys.exit(-1)


if __name__ == '__main__':
    main()
