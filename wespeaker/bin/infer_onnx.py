# Copyright (c) 2022, Shuai Wang (wsstriving@gmail.com)
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

import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


def get_args():
    parser = argparse.ArgumentParser(description='infer example using onnx')
    parser.add_argument('--onnx_path', required=True, help='onnx path')
    parser.add_argument('--wav_path', required=True, help='wav path')
    args = parser.parse_args()
    return args


def compute_fbank(wav_path,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank, simlilar to the one in wespeaker.dataset.processor,
        While integrating the wave reading and CMN.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      sample_frequency=sample_rate,
                      window_type='hamming',
                      use_energy=False)
    # CMN, without CVN
    mat = mat - torch.mean(mat, dim=0)
    return mat


def main():
    args = get_args()

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(args.onnx_path, sess_options=so)

    wav_path = args.wav_path
    feats = compute_fbank(wav_path)
    feats = feats.unsqueeze(0).numpy()  # add batch dimension

    embeddings = session.run(output_names=['embs'],
                             input_feed={'feats': feats})
    print(embeddings[0].shape)


if __name__ == '__main__':
    main()
