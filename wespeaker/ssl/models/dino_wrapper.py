# Copyright (c) Facebook, Inc. and its affiliates.
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
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
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import math
import copy

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DINOHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 use_bn=False,
                 norm_last_layer=True,
                 nlayers=3,
                 hidden_dim=2048,
                 bottleneck_dim=256,
                 normalize_input=False
                 ):
        super().__init__()
        self.normalize_input = normalize_input

        if nlayers == 0:
            self.mlp = nn.Identity()
        elif nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_mlp=False):
        if self.normalize_input:
            x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.mlp(x)
        if return_mlp:
            return x
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self,
                 out_dim,
                 n_scrops,
                 n_tcrops,
                 warmup_teacher_temp,
                 teacher_temp,
                 nepochs,
                 warmup_teacher_temp_epochs_ratio=0.2,
                 student_temp=0.1,
                 center_momentum=0.9
                 ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_scrops = n_scrops
        self.n_tcrops = n_tcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning

        warmup_teacher_temp_epochs = int(nepochs * warmup_teacher_temp_epochs_ratio)
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        self.student_entropy = 0.0
        self.teacher_entropy = 0.0

    def forward(self, student_output, teacher_output, epoch, mode=0):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_tmp = student_out.detach()
        student_out = student_out.chunk(self.n_scrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_tmp = (teacher_output.detach() - self.center) / temp
        teacher_out = teacher_out.detach().chunk(self.n_tcrops)

        student_tmp = F.softmax(student_tmp, dim=1) + 1e-7
        teacher_tmp = F.softmax(teacher_tmp, dim=1) + 1e-7
        self.student_entropy = torch.mean(torch.sum(-student_tmp * torch.log(student_tmp), dim=1)).item()
        self.teacher_entropy = torch.mean(torch.sum(-teacher_tmp * torch.log(teacher_tmp), dim=1)).item()

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if mode == 0: # skip the same
                    if v == iq:
                        continue
                elif mode == 1: # only the channel invariant
                    if v != iq:
                        continue
                elif mode == 2: # only the content invariant
                    if v < 2:
                        continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * get_world_size())
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DINO(nn.Module):
    """
    https://arxiv.org/abs/2104.14294
    """
    def __init__(self, base_model, dino_head_args, dino_loss_args, sync_bn=True):
        """
        model: the student and teacher base model
        """
        super(DINO, self).__init__()

        # get the student and teacher model
        self.s_model = base_model
        self.t_model = copy.deepcopy(base_model)

        self.s_model.add_module("projection_head", DINOHead(**dino_head_args))
        self.t_model.add_module("projection_head", DINOHead(**dino_head_args))
        self.t_model.projection_head.load_state_dict(self.s_model.projection_head.state_dict())

        if sync_bn:
            self.s_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.s_model)
            self.t_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.t_model)

        # the teacher model is not updated from back propagation
        for p in self.t_model.parameters():
            p.requires_grad = False

        # init dino loss
        self.dino_loss_calculator = DINOLoss(**dino_loss_args)

    @torch.no_grad()
    def ema_update(self, m=0.0):
        for param_q, param_k in zip(self.s_model.parameters(),
                                    self.t_model.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def forward(self, local_feats, global_feats, epoch=0):
        """
        Input:
            local_feats: (chunk_num * B, T, F)
            global_feats: (chunk_num' * B, T, F)
        Output:
            loss: a scalar value
        """
        # feed global and local features into student model
        g_outputs = self.s_model(global_feats)
        l_outputs = self.s_model(local_feats)
        g_output = g_outputs[-1] if isinstance(g_outputs, tuple) else g_outputs
        l_output = l_outputs[-1] if isinstance(l_outputs, tuple) else l_outputs
        s_output = torch.cat([g_output, l_output])
        s_output = self.s_model.projection_head(s_output)
        # feed global features into teacher model
        with torch.no_grad():
            t_outputs = self.t_model(global_feats)
            t_output = t_outputs[-1] if isinstance(t_outputs, tuple) else t_outputs
            t_output = self.t_model.projection_head(t_output)

        # compute CE loss
        loss = self.dino_loss_calculator(s_output, t_output, epoch)

        return loss
