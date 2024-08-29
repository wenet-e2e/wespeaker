# Copyright (c) 2024 Hongji Wang (jijijiang77@gmail.com)
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

import contextlib
import torch
import torch.nn as nn

import s3prl
from s3prl.nn import Featurizer, S3PRLUpstream


class S3prlFrontend(nn.Module):
    """Speech Pretrained Representation Frontend."""

    def __init__(self,
                 upstream_args: dict,
                 download_dir: str = "./s3prl_hub",
                 multilayer_feature: bool = True,
                 layer: int = -1,
                 frozen: bool = False,
                 frame_shift: int = 20,
                 frame_length: int = 20,
                 sample_rate: int = 16000):
        super().__init__()

        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.frozen = frozen

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert upstream_args.get("name",
                                 None) in S3PRLUpstream.available_names()
        self.upstream = S3PRLUpstream(
            upstream_args.get("name"),
            path_or_url=upstream_args.get("path_or_url", None),
            normalize=upstream_args.get("normalize", False),
            extra_conf=upstream_args.get("extra_conf", None),
        )
        if getattr(self.upstream.upstream, "model", None):
            if getattr(self.upstream.upstream.model, "feature_grad_mult",
                       None) is not None:
                self.upstream.upstream.model.feature_grad_mult = 1.0
        self.upstream.eval()

        if layer != -1:
            layer_selections = [layer]
            assert not multilayer_feature, \
                "multilayer_feature must be False if layer is specified"
        else:
            layer_selections = None
        self.featurizer = Featurizer(self.upstream,
                                     layer_selections=layer_selections)

        assert self.featurizer.downsample_rate == sample_rate * frame_shift // 1000

        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.upstream.named_parameters():
                if "mask_emb" in name:
                    param.requires_grad_(False)

    def output_size(self):
        return self.featurizer.output_size

    def forward(self, input: torch.Tensor, input_lengths: torch.LongTensor):
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            feats, feats_lens = self.upstream(input, input_lengths)
        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        return feats, feats_lens
