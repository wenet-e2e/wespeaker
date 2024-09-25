# Copyright (c) 2024 Shuai Wang (wsstriving@gmail.com)
#               2024 Tianchi Liu (tianchi_liu@u.nus.edu)
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
'''The implementation of Gemini-DF-ResNet.

Reference:
[1] Liu, Tianchi, et al. "Golden Gemini is All You Need: Finding the 
    Sweet Spots for Speaker Verification." arXiv:2312.03620 (2023).
[2] Liu, Bei, et al. "DF-ResNet: Boosting Speaker Verification Performance 
    with Depth-First Design." INTERSPEECH. 2022. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers


class Inverted_Bottleneck(nn.Module):
    def __init__(self, dim):
        super(Inverted_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4 * dim)
        self.conv2 = nn.Conv2d(4 * dim, 4 * dim, 
                               kernel_size=3, padding=1, groups=4 * dim, 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(4 * dim)
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = F.relu(out)
        return out


class Gemini_DF_ResNet(nn.Module):
    # DF_ResNet with T14c stride strategy of Golden Gemini
    def __init__(self,
                 depths,
                 dims,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(Gemini_DF_ResNet, self).__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8 / 2) * dims[-1]
        self.two_emb_layer = two_emb_layer

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )
        self.downsample_layers.append(stem)

        stride_f = [2, 2, 2, 2]
        stride_t = [1, 2, 1, 1]

        for i in range(4):
            downsample_layer = nn.Sequential(
                nn.Conv2d(
                    dims[i], dims[i + 1], kernel_size=3,
                    stride=(stride_f[i], stride_t[i]),
                    padding=1, bias=False),
                nn.BatchNorm2d(dims[i + 1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Inverted_Bottleneck(dim=dims[i + 1]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.pool = getattr(pooling_layers,
                            pooling_func)(in_dim=self.stats_dim)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _get_frame_level_feat(self, x):
        # for inner class usage
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = self.downsample_layers[0](x)
        out = self.downsample_layers[1](out)
        out = self.stages[0](out)
        out = self.downsample_layers[2](out)
        out = self.stages[1](out)
        out = self.downsample_layers[3](out)
        out = self.stages[2](out)
        out = self.downsample_layers[4](out)
        out = self.stages[3](out)

        return out

    def get_frame_level_feat(self, x):
        # for outer interface 
        out = self._get_frame_level_feat(x)
        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)

        return out  # (B, T, D)

    def forward(self, x):

        out = self._get_frame_level_feat(x)
        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a


# following models do include separate downsmapling layers into layer counting
def Gemini_DF_ResNet60(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return Gemini_DF_ResNet(depths=[3, 3, 9, 3],
                            dims=[32, 32, 64, 128, 256],
                            feat_dim=feat_dim,
                            embed_dim=embed_dim,
                            pooling_func=pooling_func,
                            two_emb_layer=two_emb_layer)


def Gemini_DF_ResNet114(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return Gemini_DF_ResNet(depths=[3, 3, 27, 3],
                            dims=[32, 32, 64, 128, 256],
                            feat_dim=feat_dim,
                            embed_dim=embed_dim,
                            pooling_func=pooling_func,
                            two_emb_layer=two_emb_layer)


def Gemini_DF_ResNet183(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return Gemini_DF_ResNet(depths=[3, 8, 45, 3],
                            dims=[32, 32, 64, 128, 256],
                            feat_dim=feat_dim,
                            embed_dim=embed_dim,
                            pooling_func=pooling_func,
                            two_emb_layer=two_emb_layer)


def Gemini_DF_ResNet237(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return Gemini_DF_ResNet(depths=[3, 8, 63, 3],
                            dims=[32, 32, 64, 128, 256],
                            feat_dim=feat_dim,
                            embed_dim=embed_dim,
                            pooling_func=pooling_func,
                            two_emb_layer=two_emb_layer)


if __name__ == '__main__':
    x = torch.zeros(1, 200, 80)
    model = Gemini_DF_ResNet114(80, 256, 'TSTP')
    model.eval()
    out = model(x)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))
