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

import fire
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wespeaker.dataset.dataset import FeatList_LableDict_Dataset
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.file_utils import read_scp
from wespeaker.utils.utils import parse_config_or_kwargs

from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""
    device = torch.device("cuda")
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (utts, feats, _) in tqdm(enumerate(data_loader), total=num_batches):
        feats = feats.float().to(device)  # (B,T,F)
        # Forward through model
        outputs = model(feats)  # embed or (embed_a, embed_b)

        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()


def calibrate_extract(config='conf/config.yaml', **kwargs):
    # do monkey patch to replace the linear,
    # conv modules with their quantized ones
    quant_modules.initialize()
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    calibrator = configs.get("calibrator", "max")
    percentile = None
    if calibrator == 'max':
        calib_method = 'max'
    elif calibrator == 'percentile':
        calib_method = 'histogram'
        percentile = configs.get("percentile", 99.99)
    else:
        # calibrator == 'mse' or calibrator == 'entropy'
        calib_method = 'histogram'

    quant_desc_input = QuantDescriptor(calib_method=calib_method)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConv1d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    model_path = configs['model_path']
    calibrated_model_path = configs['calibrated_model_path']
    data_scp = configs['data_scp']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)
    raw_wav = configs.get('raw_wav', True)
    feat_dim = configs['feature_args'].get('feat_dim', 80)
    num_frms = configs['feature_args'].get('num_frms', 200)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    model = get_speaker_model(configs['model'])(**configs['model_args'])
    load_checkpoint(model, model_path)
    device = torch.device("cuda")
    model.to(device).eval()

    # prepare dataset and dataloader
    data_list = read_scp(data_scp)
    dataset = FeatList_LableDict_Dataset(data_list,
                                         utt2spkid_dict={},
                                         whole_utt=(batch_size == 1),
                                         raw_wav=raw_wav,
                                         feat_dim=feat_dim,
                                         num_frms=num_frms)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    num_calib_batch = configs.get('num_calib_batch', 10)

    # enable all possible modules and quantizers
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()

    with torch.no_grad():
        collect_stats(model, dataloader, num_calib_batch)
        compute_amax(model, method=calibrator, percentile=percentile)

    # Save the model
    torch.save(model.state_dict(), calibrated_model_path)

    class Model(nn.Module):
        def __init__(self, model):
            super(Model, self).__init__()
            self.model = model

        def forward(self, feats):
            outputs = self.model(feats)  # embed or (embed_a, embed_b)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            return embeds

    model = Model(model)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    dummy_input = torch.ones(1, num_frms, feat_dim, dtype=torch.float32).cuda()
    torch.onnx.export(model, dummy_input,
                      calibrated_model_path.replace(".pt", ".onnx"),
                      do_constant_folding=True,
                      verbose=False,
                      opset_version=14,
                      input_names=['feats'],
                      output_names=['embs'],
                      dynamic_axes={'feats': {0: 'B', 1: 'T'}, 'embs': {0: 'B'}})

if __name__ == '__main__':
    fire.Fire(calibrate_extract)
