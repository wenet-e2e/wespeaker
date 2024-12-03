# Copyright (c) 2024, Chengdong Liang(liangchengdongd@qq.com)
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
import ncnn
import numpy as np

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output dir')
    args = parser.parse_args()
    return args


def test_ncnn_inference(in0, ncnn_param, ncnn_bin):
    outs = []

    with ncnn.Net() as net:
        net.load_param(ncnn_param)
        net.load_model(ncnn_bin)
        input_names = net.input_names()
        output_names = net.output_names()
        print("input_names: ", input_names)
        print("output_names: ", output_names)

        with net.create_extractor() as ex:

            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())
            _, out0 = ex.extract("out0")
            outs.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            if len(output_names) > 1:
                _, out1 = ex.extract("out1")
                outs.append(torch.from_numpy(np.array(out1)).unsqueeze(0))

    if len(outs) == 1:
        return outs[0]
    else:
        return tuple(outs)


def main():
    args = get_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    configs['model_args']['ncnn_mode'] = True
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    print(model)

    load_checkpoint(model, args.checkpoint)
    model.eval()
    # Export jit torch script model
    torch.manual_seed(0)
    x = torch.rand(1, 200, 80, dtype=torch.float).contiguous()
    script_model = torch.jit.trace(model, x)
    os.makedirs(args.output_dir, exist_ok=True)
    model_trace_path = os.path.join(args.output_dir, 'model.trace.pt')
    script_model.save(model_trace_path)
    print('Export trace model successfully, see {}'.format(model_trace_path))

    os.system("pnnx {} inputshape=[1,200,80]f32".format(model_trace_path))
    print('The ncnn model is saved in {} and {}'.format(
        model_trace_path[:-3] + '.ncnn.param',
        model_trace_path[:-3] + '.ncnn.bin'))

    torch_output = model(x)
    ncnn_output = test_ncnn_inference(x, model_trace_path[:-3] + '.ncnn.param',
                                      model_trace_path[:-3] + '.ncnn.bin')
    if isinstance(torch_output, tuple):
        torch_output = torch_output[1]
        ncnn_output = ncnn_output[1]

    if np.allclose(torch_output.detach().numpy(),
                   ncnn_output.detach().numpy(),
                   rtol=1e-5,
                   atol=1e-2):
        print("Export ncnn model successfully, "
              "and the output accuracy check passed!")
    else:
        print("Export ncnn model successfully, but ncnn and torchscript have "
              "different outputs when given the same input, please check!")


if __name__ == '__main__':
    main()
