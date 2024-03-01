# Copyright (c) 2024 Hongji Wang (jijijiang77@gmail.com)
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
'''
This implementation is adapted from github repo:
https://github.com/alibaba-damo-academy/3D-Speaker

Some modifications:
1. Reuse the pooling layers (small update) in wespeaker
2. Refine BasicBlockERes2Net and BasicBlockERes2Net_diff_AFF to meet
   the torch.jit.script export requirements

ERes2Net incorporates both local and global feature fusion techniques
to improve the performance. The local feature fusion (LFF) fuses the
features within one single residual block to extract the local signal.
The global feature fusion (GFF) takes acoustic features of different
scales as input to aggregate global signal. Parameters expansion,
baseWidth, and scale can be modified to obtain optimal performance.

Reference:
[1] Yafeng Chen, Siqi Zheng, Hui Wang, Luyao Cheng, Qian Chen, Jiajun Qi.
    "An Enhanced Res2Net with Local and Global Feature Fusion for Speaker
    Verification". arXiv preprint arXiv:2305.12838 (2023).
'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0.0, 20.0, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2,
                      inter_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels,
                      channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x, ds_y):
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)

        return xo


class BasicBlockERes2Net(nn.Module):

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 baseWidth=32,
                 scale=2,
                 expansion=2):
        super(BasicBlockERes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = self.relu(bn(sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2Net_diff_AFF(nn.Module):

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 baseWidth=32,
                 scale=2,
                 expansion=2):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion

        # to meet the torch.jit.script export requirements
        self.conv2_1 = conv3x3(width, width)
        self.bn2_1 = nn.BatchNorm2d(width)
        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums - 1):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
            fuse_models.append(AFF(channels=width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        sp = self.conv2_1(sp)
        sp = self.relu(self.bn2_1(sp))
        out = sp
        for i, (conv, bn, fuse_model) in enumerate(
                zip(self.convs, self.bns, self.fuse_models), 1):
            sp = fuse_model(sp, spx[i])
            sp = conv(sp)
            sp = self.relu(bn(sp))
            out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2Net(nn.Module):

    def __init__(self,
                 m_channels,
                 num_blocks,
                 baseWidth=32,
                 scale=2,
                 expansion=2,
                 block=BasicBlockERes2Net,
                 block_fuse=BasicBlockERes2Net_diff_AFF,
                 feat_dim=80,
                 embed_dim=192,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(ERes2Net, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer
        self.expansion = expansion

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1,
                                       baseWidth=baseWidth,
                                       scale=scale,
                                       expansion=expansion)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2,
                                       baseWidth=baseWidth,
                                       scale=scale,
                                       expansion=expansion)
        self.layer3 = self._make_layer(block_fuse,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2,
                                       baseWidth=baseWidth,
                                       scale=scale,
                                       expansion=expansion)
        self.layer4 = self._make_layer(block_fuse,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2,
                                       baseWidth=baseWidth,
                                       scale=scale,
                                       expansion=expansion)

        # Downsampling module for each layer
        self.layer1_downsample = nn.Conv2d(m_channels * expansion,
                                           m_channels * expansion * 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * expansion * 2,
                                           m_channels * expansion * 4,
                                           kernel_size=3,
                                           padding=1,
                                           stride=2,
                                           bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * expansion * 4,
                                           m_channels * expansion * 8,
                                           kernel_size=3,
                                           padding=1,
                                           stride=2,
                                           bias=False)

        # Bottom-up fusion module
        self.fuse_mode12 = AFF(channels=m_channels * expansion * 2)
        self.fuse_mode123 = AFF(channels=m_channels * expansion * 4)
        self.fuse_mode1234 = AFF(channels=m_channels * expansion * 8)

        self.pool = getattr(pooling_layers,
                            pooling_func)(in_dim=self.stats_dim * expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self,
                    block,
                    planes,
                    num_blocks,
                    stride,
                    baseWidth=32,
                    scale=2,
                    expansion=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, baseWidth, scale,
                      expansion))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        stats = self.pool(fuse_out1234)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


def ERes2Net34_Base(feat_dim,
                    embed_dim,
                    pooling_func='TSTP',
                    two_emb_layer=False):
    return ERes2Net(32, [3, 4, 6, 3],
                    feat_dim=feat_dim,
                    embed_dim=embed_dim,
                    pooling_func=pooling_func,
                    two_emb_layer=two_emb_layer)


def ERes2Net34_Large(feat_dim,
                     embed_dim,
                     pooling_func='TSTP',
                     two_emb_layer=False):
    return ERes2Net(64, [3, 4, 6, 3],
                    feat_dim=feat_dim,
                    embed_dim=embed_dim,
                    pooling_func=pooling_func,
                    two_emb_layer=two_emb_layer)


def ERes2Net34_aug(feat_dim,
                   embed_dim,
                   pooling_func='TSTP',
                   two_emb_layer=False,
                   expansion=4,
                   baseWidth=24,
                   scale=3):
    return ERes2Net(64, [3, 4, 6, 3],
                    expansion=expansion,
                    baseWidth=baseWidth,
                    scale=scale,
                    feat_dim=feat_dim,
                    embed_dim=embed_dim,
                    pooling_func=pooling_func,
                    two_emb_layer=two_emb_layer)


if __name__ == '__main__':
    x = torch.zeros(1, 200, 80)
    model = ERes2Net34_Base(feat_dim=80, embed_dim=512, two_emb_layer=False)
    model.eval()
    out = model(x)
    print(out.size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
