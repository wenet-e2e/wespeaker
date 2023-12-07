# Copyright (c) 2022, NVIDIA CORPORATION.
#                     Shuai Wang (wsstriving@gmail.com)
#                     Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
# All rights reserved.
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

from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn as nn
import yaml
import os

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_model', required=True, help='output file')
    parser.add_argument('--mean_vec',
                        required=False,
                        default=None,
                        help='mean vector')
    # NOTE(cdliang): for horizon bpu, the shape of input is fixed.
    parser.add_argument('--num_frames',
                        type=int,
                        required=True,
                        help="num frames")
    args = parser.parse_args()
    return args


class Model(nn.Module):

    def __init__(self, model, mean_vec=None):
        super(Model, self).__init__()
        self.model = model
        self.register_buffer("mean_vec", mean_vec)

    def forward(self, feats):
        # NOTE(cdliang): for horizion x3pi, input shape is [NHWC]
        feats = feats.squeeze(1)  # [B, 1, T, F] -> [B, T, F]
        outputs = self.model(feats)  # embed or (embed_a, embed_b)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        embeds = embeds - self.mean_vec
        return embeds


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = get_speaker_model(configs['model'])(**configs['model_args'])
    load_checkpoint(model, args.checkpoint)
    model.eval()

    if args.mean_vec:
        mean_vec = torch.tensor(np.load(args.mean_vec), dtype=torch.float32)
    else:
        embed_dim = configs["model_args"]["embed_dim"]
        mean_vec = torch.zeros(embed_dim, dtype=torch.float32)

    model = Model(model, mean_vec)
    model.eval()

    feat_dim = configs['model_args'].get('feat_dim', 80)
    static_input = torch.ones(1, 1, args.num_frames, feat_dim)
    torch.onnx.export(model,
                      static_input,
                      args.output_model,
                      do_constant_folding=True,
                      verbose=False,
                      opset_version=11,
                      input_names=['feats'],
                      output_names=['embs'])


if __name__ == '__main__':
    main()
