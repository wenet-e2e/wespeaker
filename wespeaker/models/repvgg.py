# Copyright (c) 2021 xmuspeech (Author: Leo)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2024 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some modifications from the original architecture:
REPVGG_TINY_A0: Smaller hidden_dim and Deeper structure

Ref:
1. RepVGG: Making VGG-style ConvNets Great Again
   (https://arxiv.org/pdf/2101.03697)
   Github: https://github.com/DingXiaoH/RepVGG
2. Rep Works in Speaker Verification (https://arxiv.org/pdf/2110.09720)
3. asv-subtools:
   https://github.com/Snowdar/asv-subtools/blob/master/pytorch/libs/nnet/repvgg.py
"""

import torch.nn as nn
import numpy as np
import torch
import copy
import wespeaker.models.pooling_layers as pooling_layers

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = dict.fromkeys(optional_groupwise_layers, 2)
g4_map = dict.fromkeys(optional_groupwise_layers, 4)


class SEBlock_2D(torch.nn.Module):
    """ A SE Block layer layer which can learn to use global information to
    selectively emphasise informative features and suppress less useful ones.
    This is a pytorch implementation of SE Block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
        leo 2020-12-20 [Check and update]
        """

    def __init__(self, in_planes, ratio=16, inplace=True):
        '''
        @ratio: a reduction ratio which allows us to vary the capacity
        and computational cost of the SE blocks
        in the network.
        '''
        super(SEBlock_2D, self).__init__()

        self.in_planes = in_planes
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_1 = torch.nn.Linear(in_planes, in_planes // ratio)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.fc_2 = torch.nn.Linear(in_planes // ratio, in_planes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch),
                 including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 4
        assert inputs.shape[1] == self.in_planes

        b, c, _, _ = inputs.size()
        x = self.avg_pool(inputs).view(b, c)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)

        scale = x.view(b, c, 1, 1)
        return inputs * scale


def conv_bn(in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            groups=1):
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU(inplace=True)

        if use_se:
            self.se = SEBlock_2D(out_channels, 4)

        else:
            self.se = nn.Identity()
        self.rbr_reparam = None
        self.rbr_identity = None
        self.rbr_dense = None
        self.rbr_1x1 = None
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels
            ) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)

    def forward(self, inputs):
        if self.deploy and self.rbr_reparam is not None:

            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.rbr_dense is not None and self.rbr_1x1 is not None:
            return self.se(
                self.nonlinearity(
                    self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        else:
            raise TypeError(
                "It's a training repvgg structure but branch conv not exits.")

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight
    #       and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight /
              ((self.rbr_dense.bn.running_var +
                self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var +
                                         self.rbr_1x1.bn.eps).sqrt())).reshape(
                                             -1, 1, 1, 1).detach()

        # The L2 loss of the "circle" of weights in 3x3 kernel.
        # Use regular L2 on them.
        l2_loss_circle = (K3**2).sum() - (K3[:, :, 1:2, 1:2]**2).sum()
        # The equivalent resultant central point of 3x3 kernel.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        # Normalize for an L2 coefficient comparable to regular L2.
        l2_loss_eq_kernel = (eq_kernel**2 / (t3**2 + t1**2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way
    #   You can get the equivalent kernel and bias
    #   at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training,
    #   just like you do to the other models.
    #   May be useful for quantization or pruning.

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.rbr_reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.rbr_dense = None
        self.rbr_1x1 = None
        if hasattr(self, 'rbr_identity'):
            self.rbr_identity = None
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepSPKBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 branch_dilation=2,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 use_se=False):
        super(RepSPKBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        # NOTE: RSBB block
        # ref: https://arxiv.org/pdf/2110.09720.pdf
        assert kernel_size == 3
        assert padding == 1
        assert dilation == 1
        assert branch_dilation == 2
        self.branch_dilation = branch_dilation
        self.depoly_kernel_size = (kernel_size - 1) * (branch_dilation -
                                                       1) + kernel_size

        self.nonlinearity = nn.ReLU(inplace=True)

        if use_se:
            self.se = SEBlock_2D(out_channels, 4)

        else:
            self.se = nn.Identity()
        self.rbr_reparam = None
        self.rbr_identity = None
        self.rbr_dense = None
        self.rbr_dense_dilation = None
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=self.depoly_kernel_size,
                                         stride=stride,
                                         padding=self.branch_dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels
            ) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_dense_dilation = conv_bn(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=self.branch_dilation,
                                              dilation=self.branch_dilation,
                                              groups=groups)

    def forward(self, inputs):
        if self.deploy and self.rbr_reparam is not None:

            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.rbr_dense is not None and self.rbr_dense_dilation is not None:
            return self.se(
                self.nonlinearity(
                    self.rbr_dense(inputs) + self.rbr_dense_dilation(inputs) +
                    id_out))
        else:
            raise TypeError(
                "It's a training repvgg structure but branch conv not exits.")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel_dilation_branch, bias_dilation_branch = self._fuse_bn_tensor(
            self.rbr_dense_dilation)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return self._convert_3x3_dilation_to_5x5_tensor(
            kernel_dilation_branch) + self._pad_3x3_to_5x5_tensor(
                kernel3x3) + kernelid, bias3x3 + bias_dilation_branch + biasid

    def _pad_3x3_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel3x3, [1, 1, 1, 1])

    def _convert_3x3_dilation_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            kernel_value = torch.zeros(
                (kernel3x3.size(0), kernel3x3.size(1), 5, 5),
                dtype=kernel3x3.dtype)
            kernel_value[:, :, ::2, ::2] = kernel3x3
            return kernel_value

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 5, 5),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 2, 2] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.rbr_reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.depoly_kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.branch_dilation,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.rbr_dense = None
        self.rbr_dense_dilation = None
        if hasattr(self, 'rbr_identity'):
            self.rbr_identity = None
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGG(nn.Module):

    def __init__(self,
                 head_inplanes=1,
                 block="RepVGG",
                 num_blocks=None,
                 strides=None,
                 base_width=64,
                 width_multiplier=None,
                 override_groups_map=None,
                 deploy=False,
                 use_se=False,
                 pooling_func='ASTP',
                 feat_dim=80,
                 embed_dim=256):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4
        assert len(num_blocks) == 4
        assert len(strides) == 5
        width_multiplier = [w * (base_width / 64.) for w in width_multiplier]
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se
        self.downsample_multiple = 1
        if block == "RepVGG":
            used_block = RepVGGBlock
        elif block == "RepSPK":
            used_block = RepSPKBlock
        else:
            raise TypeError("Do not support {} block.".format(block))

        for s in strides:
            self.downsample_multiple *= s

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = used_block(head_inplanes,
                                 out_channels=self.in_planes,
                                 kernel_size=3,
                                 stride=strides[0],
                                 padding=1,
                                 deploy=self.deploy,
                                 use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(used_block,
                                       int(64 * width_multiplier[0]),
                                       num_blocks[0],
                                       stride=strides[1])
        self.stage2 = self._make_stage(used_block,
                                       int(128 * width_multiplier[1]),
                                       num_blocks[1],
                                       stride=strides[2])
        self.stage3 = self._make_stage(used_block,
                                       int(256 * width_multiplier[2]),
                                       num_blocks[2],
                                       stride=strides[3])
        self.stage4 = self._make_stage(used_block,
                                       int(512 * width_multiplier[3]),
                                       num_blocks[3],
                                       stride=strides[4])
        self.output_planes = self.in_planes
        self.stats_dim = self.output_planes * int(feat_dim / 8)

        self.pool = getattr(pooling_layers,
                            pooling_func)(in_dim=self.stats_dim)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg = nn.Linear(self.pool_out_dim, embed_dim)

        # init paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0., 0.01)
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                block(in_channels=self.in_planes,
                      out_channels=planes,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=cur_groups,
                      deploy=self.deploy,
                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def get_downsample_multiple(self):
        return self.downsample_multiple

    def get_output_planes(self):
        return self.output_planes

    def _get_frame_level_feat(self, x):
        # for inner class usage
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)
        x = x.unsqueeze_(1)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x

    def get_frame_level_feat(self, x):
        # for outer interface
        out = self._get_frame_level_feat(x)
        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)

        return out  # (B, T, D)

    def forward(self, x):
        x = self._get_frame_level_feat(x)
        stats = self.pool(x)
        embed = self.seg(stats)

        return embed


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


# NOTE(cdliang): REPVGG_TINY_A0: Smaller hidden_dim and Deeper structure
def REPVGG_TINY_A0(feat_dim,
                   embed_dim,
                   pooling_func='TSTP',
                   deploy=False,
                   use_se=False):
    return RepVGG(num_blocks=[3, 4, 23, 3],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[0.5, 0.5, 0.5, 0.5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_TINY_RSBB_A0(feat_dim,
                        embed_dim,
                        pooling_func='TSTP',
                        deploy=False,
                        use_se=False):
    return RepVGG(num_blocks=[3, 4, 23, 3],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[0.5, 0.5, 0.5, 0.5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim,
                  block='RepSPK')


def REPVGG_A0(feat_dim,
              embed_dim,
              pooling_func='TSTP',
              deploy=False,
              use_se=False):
    return RepVGG(num_blocks=[2, 4, 14, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[0.75, 0.75, 0.75, 2.5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_RSBB_A0(feat_dim,
                   embed_dim,
                   pooling_func='TSTP',
                   deploy=False,
                   use_se=False):
    return RepVGG(num_blocks=[2, 4, 14, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[0.75, 0.75, 0.75, 2.5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim,
                  block='RepSPK')


def REPVGG_A1(feat_dim,
              embed_dim,
              pooling_func='TSTP',
              deploy=False,
              use_se=False):
    return RepVGG(num_blocks=[2, 4, 14, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[1, 1, 1, 2.5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_A2(feat_dim,
              embed_dim,
              pooling_func='TSTP',
              deploy=False,
              use_se=False):
    return RepVGG(num_blocks=[2, 4, 14, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[1.5, 1.5, 1.5, 2.75],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_RSBB_A2(feat_dim,
                   embed_dim,
                   pooling_func='TSTP',
                   deploy=False,
                   use_se=False):
    return RepVGG(num_blocks=[2, 4, 14, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[1.5, 1.5, 1.5, 2.75],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim,
                  block='RepSPK')


def REPVGG_B0(feat_dim,
              embed_dim,
              pooling_func='TSTP',
              deploy=False,
              use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[1, 1, 1, 2.5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_RSBB_B0(feat_dim,
                   embed_dim,
                   pooling_func='TSTP',
                   deploy=False,
                   use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[1, 1, 1, 2.5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim,
                  block='RepSPK')


def REPVGG_B1(feat_dim,
              embed_dim,
              pooling_func='TSTP',
              deploy=False,
              use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[2, 2, 2, 4],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_B1g2(feat_dim,
                embed_dim,
                pooling_func='TSTP',
                deploy=False,
                use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[2, 2, 2, 4],
                  override_groups_map=g2_map,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_B1g4(feat_dim,
                embed_dim,
                pooling_func='TSTP',
                deploy=False,
                use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[2, 2, 2, 4],
                  override_groups_map=g4_map,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_B2(feat_dim,
              embed_dim,
              pooling_func='TSTP',
              deploy=False,
              use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[2.5, 2.5, 2.5, 5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_B2g2(feat_dim,
                embed_dim,
                pooling_func='TSTP',
                deploy=False,
                use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[2.5, 2.5, 2.5, 5],
                  override_groups_map=g2_map,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_B2g4(feat_dim,
                embed_dim,
                pooling_func='TSTP',
                deploy=False,
                use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[2.5, 2.5, 2.5, 5],
                  override_groups_map=g4_map,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_B3(feat_dim,
              embed_dim,
              pooling_func='TSTP',
              deploy=False,
              use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[3, 3, 3, 5],
                  override_groups_map=None,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_B3g2(feat_dim,
                embed_dim,
                pooling_func='TSTP',
                deploy=False,
                use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[3, 3, 3, 5],
                  override_groups_map=g2_map,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_B3g4(feat_dim,
                embed_dim,
                pooling_func='TSTP',
                deploy=False,
                use_se=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[3, 3, 3, 5],
                  override_groups_map=g4_map,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


def REPVGG_D2SE(feat_dim,
                embed_dim,
                pooling_func='TSTP',
                deploy=False,
                use_se=True):
    return RepVGG(num_blocks=[8, 14, 24, 1],
                  strides=[1, 1, 2, 2, 2],
                  width_multiplier=[2.5, 2.5, 2.5, 5],
                  override_groups_map=g4_map,
                  deploy=deploy,
                  use_se=use_se,
                  pooling_func=pooling_func,
                  embed_dim=embed_dim,
                  feat_dim=feat_dim)


if __name__ == '__main__':
    x = torch.zeros(1, 200, 80)
    model = REPVGG_TINY_A0(feat_dim=80,
                           embed_dim=256,
                           pooling_func='TSTP',
                           deploy=True,
                           use_se=False)
    model.eval()
    y = model(x)
    print(y.size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
