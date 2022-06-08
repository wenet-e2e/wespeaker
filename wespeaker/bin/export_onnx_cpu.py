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

from __future__ import print_function

import argparse
import os
import torch
import yaml

from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.models.speaker_model import get_speaker_model


def get_args():
    parser = argparse.ArgumentParser(description='export your wespeaker model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output directory')
    args = parser.parse_args()
    return args


def export_onnx(model, feat_dim, onnx_path):
    input_names = [
        'x',
    ]
    output_names = [
        'embed_a',
        'embed_b',  # We usually use this one for better performance
    ]
    dynamic_axes = {
        'x': {0: 'batch_size', 1: 'num_frames'},
    }

    dummy_input = torch.rand([1, 200, feat_dim])

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes)


def main():
    args = get_args()
    os.system("mkdir -p " + args.output_dir)

    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = get_speaker_model(configs['model'])(**configs['model_args'])
    feat_dim = configs['feature_args'].get('feat_dim', 80)

    load_checkpoint(model, args.checkpoint)

    # Export onnx cpu model
    onnx_outpath = os.path.join(args.output_dir, 'model.onnx')
    export_onnx(model, feat_dim, onnx_outpath)
    print('Export model successfully, see {}'.format(onnx_outpath))


if __name__ == '__main__':
    main()
