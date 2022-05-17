# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function

import argparse
import os

import torch
import yaml

from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.quant_helper import set_quantizer_by_name
import torch.nn as nn
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_model', required=True, help='output file')
    parser.add_argument('--mean_vec', required=False, default=None, help='mean vector')
    parser.add_argument('--quantized',
                        required=False,
                        action="store_true",
                        help="whether the model is quantized")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    if args.quantized:
        # do monkey patch to replace the linear,
        # conv modules with their quantized ones
        quant_modules.initialize()
        print("QUANT MODULES INITIALIZED")

    model = get_speaker_model(configs['model'])(**configs['model_args'])

    load_checkpoint(model, args.checkpoint)

    if not args.quantized:
        set_quantizer_by_name(model, [''], _disabled=True)

    if args.mean_vec:
        mean_vec = torch.tensor(np.load(args.mean_vec), dtype=torch.float32)
    else:
        embed_dim = configs["model_args"]["embed_dim"]
        mean_vec = torch.zeros(embed_dim, dtype=torch.float32)

    class Model(nn.Module):
        def __init__(self, model, mean_vec=None):
            super(Model, self).__init__()
            self.model = model
            self.register_buffer("mean_vec", mean_vec)

        def forward(self, feats):
            outputs = self.model(feats)  # embed or (embed_a, embed_b)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            embeds = embeds - self.mean_vec
            return embeds

    model = Model(model, mean_vec)
    model.eval()
    feat_dim = configs['feature_args'].get('feat_dim', 80)
    num_frms = configs['feature_args'].get('num_frms', 200)

    if args.quantized:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

    dummy_input = torch.ones(1, num_frms, feat_dim)
    torch.onnx.export(
        model, dummy_input,
        args.output_model,
        do_constant_folding=True,
        verbose=False,
        opset_version=14,
        input_names=['feats'],
        output_names=['embs'],
        dynamic_axes={'feats': {0: 'B', 1: 'T'}, 'embs': {0: 'B'}})

    # You may further generate tensorrt engine:
    # trtexec --onnx=avg_model.onnx --minShapes=feats:1x200x80 \
    # --optShapes=feats:64x200x80 --maxShapes=feats:128x200x80 \
    # --fp16
    # Notice T = 200 is not a must, you may change it to other size:
    # trtexec --onnx=avg_model.onnx --minShapes=feats:1x100x80 \
    # --optShapes=feats:64x200x80 --maxShapes=feats:128x500x80 \
    # --fp16
    # If it is an model with QDQ nodes, please add --int8  

if __name__ == '__main__':
    main()
