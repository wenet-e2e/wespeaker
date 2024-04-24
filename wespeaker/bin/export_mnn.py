# Copyright (c) 2024, Chengdong Liang(liangchengdongd@qq.com)
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

import os
import argparse

import numpy as np
import torch
import MNN
import onnxruntime as ort


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--onnx_model', required=True, help='onnx model')
    parser.add_argument('--output_model', required=True, help='output file')
    parser.add_argument('--num_frames',
                        default=-1,
                        type=int,
                        help='fix number of frames')
    args = parser.parse_args()
    return args


def test_onnx_inference(in0, model_path):
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options=so)
    output = session.run(output_names=['embs'], input_feed={'feats': in0})
    return output[0]


def test_mnn_inference(in0, model_path):
    config = {}
    config["precision"] = "high"
    config["backend"] = 0
    config["numThread"] = 1

    rt = MNN.nn.create_runtime_manager((config, ))
    net = MNN.nn.load_module_from_file(model_path, ["feats"], ["embs"],
                                       runtime_manager=rt)

    input_tensor = MNN.expr.convert(in0, MNN.expr.NC4HW4)
    output_tensor = net.forward(input_tensor)
    output_tensor = MNN.expr.convert(output_tensor, MNN.expr.NCHW)
    output = output_tensor.read()
    return output


def main():
    args = get_args()
    # 1. convert onnx to mnn
    if args.num_frames > 0:
        os.system(
            "MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode MNN \
            --saveStaticModel".format(args.onnx_model, args.output_model))
    else:
        os.system(
            "MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode MNN".
            format(args.onnx_model, args.output_model))
    print("Exported MNN model to ", args.output_model)
    # 2. print model info
    os.system("MNNConvert -f MNN --modelFile {} --info".format(
        args.output_model))
    # 3. check precision
    torch.manual_seed(0)
    if args.num_frames > 0:
        in0 = torch.rand(1, args.num_frames, 80, dtype=torch.float)
    else:
        in0 = torch.rand(1, 200, 80, dtype=torch.float)
    mnn_out = test_mnn_inference(in0.numpy(), args.output_model)
    onnx_out = test_onnx_inference(in0.numpy(), args.onnx_model)
    if np.allclose(onnx_out, mnn_out, rtol=1e-05, atol=1e-02):
        print("Export mnn model successfully, "
              "and the output accuracy check passed!")
    else:
        print("Export mnn model successfully, but onnx and mnn have different"
              " outputs when given the same input, please check!")


if __name__ == "__main__":
    main()
