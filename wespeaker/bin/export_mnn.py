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
import logging

import numpy as np
import torch
import MNN
import onnxruntime as ort

from wespeaker.bin.export_onnx import export_onnx

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_model', required=True, help='output file')
    parser.add_argument('--mean_vec',
                        required=False,
                        default=None,
                        help='mean vector')
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

    # input_var = np.expand_dims(in0.numpy(), 0)
    input_var = MNN.expr.convert(in0, MNN.expr.NC4HW4)
    output_var = net.forward(input_var)

    output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)

    output_var = output_var.read()

    return output_var


def export_mnn(checkpoint_path, config_path, output_model, mean_vec):
    # 1. export onnx
    export_onnx(checkpoint_path, config_path, output_model + ".onnx", mean_vec)
    # 2. convert onnx to mnn
    os.system(
        "MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode MNN".format(
            output_model + ".onnx", output_model))
    logger.info("Exported MNN model to %s", output_model)
    # 3. print model info
    os.system("MNNConvert -f MNN --modelFile {} --info".format(output_model))
    # 4. check precision
    torch.manual_seed(0)
    in0 = torch.rand(1, 200, 80, dtype=torch.float)
    mnn_out = test_mnn_inference(in0.numpy(), output_model)
    onnx_out = test_onnx_inference(in0.numpy(), output_model + ".onnx")
    np.testing.assert_allclose(onnx_out, mnn_out, rtol=1e-05, atol=1e-02)
    logger.info(
        "The output results of the mnn model and onnx model are consistent")


if __name__ == "__main__":
    args = get_args()
    export_mnn(args.checkpoint, args.config, args.output_model, args.mean_vec)
