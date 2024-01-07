# Copyright (c) 2022 Xu Xiang
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
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
import argparse
import kaldiio
from collections import OrderedDict

from tqdm import tqdm

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wespeaker.utils.utils import validate_path


def read_scp(scp_file):
    utt_to_wav = OrderedDict()
    for line in open(scp_file, 'r'):
        utt, wav = line.strip().split()
        utt_to_wav[utt] = wav

    return utt_to_wav


def read_segments(segments_file):
    utt_to_segments = OrderedDict()
    for line in open(segments_file, 'r'):
        seg, utt, begin, end = line.strip().split()
        begin, end = float(begin), float(end)
        if utt not in utt_to_segments:
            utt_to_segments[utt] = [(seg, begin, end)]
        else:
            utt_to_segments[utt].append((seg, begin, end))

    return utt_to_segments


def get_speech_segments(utt_to_wav, utt_to_segments):
    speech_segments_id = []
    speech_segments = []

    for utt, wav_path in utt_to_wav.items():
        segments = utt_to_segments[utt]
        signal, sr = torchaudio.load(wav_path)
        signal = signal.squeeze()

        for seg, begin, end in segments:
            speech_segments_id.append(seg)
            speech_segments.append(signal[int(begin * sr):int(end * sr)])

    return speech_segments_id, speech_segments


def compute_fbank(wav,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  sample_frequency=16000,
                  subseg_cmn=True):

    wav = wav.unsqueeze(0) * (1 << 15)
    feat = kaldi.fbank(wav,
                       num_mel_bins=num_mel_bins,
                       frame_length=frame_length,
                       frame_shift=frame_shift,
                       dither=dither,
                       sample_frequency=sample_frequency,
                       window_type='hamming',
                       use_energy=False)
    if not subseg_cmn:
        feat = feat - torch.mean(feat, dim=0)  # CMN

    return feat.cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--scp', required=True, help='wav scp')
    parser.add_argument('--segments', required=True, help='vad segments')
    parser.add_argument('--ark-path',
                        required=True,
                        help='path to store feat ark')
    parser.add_argument('--subseg-cmn',
                        default=True,
                        type=lambda x: x.lower() == 'true',
                        help='do cmn after or before fbank sub-segmentation')
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    utt_to_wav = read_scp(args.scp)
    utt_to_segments = read_segments(args.segments)
    speech_segments_id, speech_segments = get_speech_segments(
        utt_to_wav, utt_to_segments)

    validate_path(args.ark_path)
    feat_ark = os.path.abspath(args.ark_path)
    feat_scp = feat_ark[:-3] + "scp"

    with kaldiio.WriteHelper('ark,scp:' + feat_ark + "," + feat_scp) as writer:
        for i, speech_seg in enumerate(tqdm(speech_segments)):
            fbank_feat = compute_fbank(speech_seg, subseg_cmn=args.subseg_cmn)
            writer(speech_segments_id[i], fbank_feat)


if __name__ == '__main__':
    main()
