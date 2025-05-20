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

import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack, from_dlpack
import torch
import kaldifeat
from typing import List
import json


class Fbank(torch.nn.Module):

    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        feats = self.fbank(waves)
        B, T, F = len(feats), feats[0].size(0), feats[0].size(1)
        feats = torch.cat(feats, axis=0)
        feats = torch.reshape(feats, (B, T, F))
        feats = feats - torch.mean(feats, dim=1, keepdim=True)
        return feats


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance
          *                           device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        params = self.model_config['parameters']
        opts = kaldifeat.FbankOptions()
        opts.frame_opts.window_type = 'hamming'
        opts.frame_opts.dither = 1  # 0 -> 1
        opts.htk_compat = True

        for li in params.items():
            key, value = li
            value = value["string_value"]
            if key == "num_mel_bins":
                opts.mel_opts.num_bins = int(value)
            elif key == "frame_shift_in_ms":
                opts.frame_opts.frame_shift_ms = float(value)
            elif key == "frame_length_in_ms":
                opts.frame_opts.frame_length_ms = float(value)
            elif key == "sample_rate":
                opts.frame_opts.samp_freq = int(value)
        opts.device = torch.device(self.device)
        self.opts = opts
        self.feature_extractor = Fbank(self.opts)
        self.feature_size = opts.mel_opts.num_bins

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse.
          The length of this list must be the same as `requests`
        """
        batch_count = []
        total_waves = []

        responses = []
        for request in requests:
            # the requests will all have the same shape
            # different shape request will be
            # separated by triton inference server
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            cur_b_wav = from_dlpack(input0.to_dlpack())
            cur_b_wav = cur_b_wav * (1 << 15)  # b x -1
            cur_batch = cur_b_wav.shape[0]
            batch_count.append(cur_batch)

            for wav in cur_b_wav:
                total_waves.append(wav.to(self.device))

        features = self.feature_extractor(total_waves).cpu()
        idx = 0
        for b in batch_count:
            batch_speech = features[idx:idx + b]
            idx += b
            out0 = pb_utils.Tensor.from_dlpack("speech",
                                               to_dlpack(batch_speech))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0])
            responses.append(inference_response)
        return responses
