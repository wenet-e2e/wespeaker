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

import random
import torch


def apply_cmvn(feats, norm_mean=True, norm_var=False):
    # feats batch: (B,T,F)
    if norm_mean:
        feats = feats - torch.mean(feats, dim=1, keepdim=True)
    if norm_var:
        feats = feats / torch.sqrt(torch.var(feats, dim=1, keepdim=True) + 1e-7)

    return feats


def spec_aug(feats, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6):
    # feats batch: (B,T,F)
    # do spec_aug on all batch samples using a same group of params randomly
    # TODO (hongji): do spec_aug on each sample separately
    if random.random() < prob:
        x = feats
        assert isinstance(x, torch.Tensor)
        # y = x.clone().detach()
        y = x.detach()  # inplace operation
        _, max_frames, max_freq = y.shape
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[:, start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, :, start:end] = 0
        feats = y

    return feats
