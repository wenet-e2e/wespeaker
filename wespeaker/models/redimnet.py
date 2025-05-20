# Copyright (c) 2024 https://github.com/IDRnD/ReDimNet
#               2024 Shuai Wang (wsstriving@gmail.com)
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
"""Redimnet in pytorch.

Reference:
Paper: "Reshape Dimensions Network for Speaker Recognition"
Repo: https://github.com/IDRnD/ReDimNet

Cite:
@misc{yakovlev2024reshapedimensionsnetworkspeaker,
      title={Reshape Dimensions Network for Speaker Recognition},
      author={Ivan Yakovlev and Rostislav Makarov and Andrei Balykin
      and Pavel Malov and Anton Okhotnikov and Nikita Torgashov},
      year={2024},
      eprint={2407.18223},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2407.18223},
}
"""
import math

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers

MaxPoolNd = {1: nn.MaxPool1d, 2: nn.MaxPool2d}
ConvNd = {1: nn.Conv1d, 2: nn.Conv2d}
BatchNormNd = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d}


class to1d(nn.Module):

    def forward(self, x):
        size = x.size()
        bs, c, f, t = tuple(size)
        return x.permute((0, 2, 1, 3)).reshape((bs, c * f, t))


class NewGELUActivation(nn.Module):

    def forward(self, input):
        return (0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) *
            (input + 0.044715 * torch.pow(input, 3.0)))))


class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last or channels_first.
    The ordering of the dimensions in the inputs.
    channels_last corresponds to inputs with shape (batch_size, T, channels)
    while channels_first corresponds to shape (batch_size, channels, T).
    """

    def __init__(self, C, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.C = (C, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.C, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)

            w = self.weight
            b = self.bias
            for _ in range(x.ndim - 2):
                w = w.unsqueeze(-1)
                b = b.unsqueeze(-1)
            x = w * x + b
            return x

    def extra_repr(self) -> str:
        return ", ".join([
            f"{k}={v}" for k, v in {
                "C": self.C,
                "data_format": self.data_format,
                "eps": self.eps,
            }.items()
        ])


class GRU(nn.Module):

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__()
        self.gru = nn.GRU(*args, **kwargs)

    def forward(self, x):
        # x : (bs,C,T)
        return self.gru(x.permute((0, 2, 1)))[0].permute((0, 2, 1))


class PosEncConv(nn.Module):

    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(C,
                              C,
                              ks,
                              padding=ks // 2,
                              groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return x + self.norm(self.conv(x))


class ConvNeXtLikeBlock(nn.Module):

    def __init__(
            self,
            C,
            dim=2,
            kernel_sizes=((3, 3), ),
            group_divisor=1,
            padding="same",
    ):
        super().__init__()
        self.dwconvs = nn.ModuleList(modules=[
            ConvNd[dim](
                C,
                C,
                kernel_size=ks,
                padding=padding,
                groups=C // group_divisor if group_divisor is not None else 1,
            ) for ks in kernel_sizes
        ])
        self.norm = BatchNormNd[dim](C * len(kernel_sizes))
        self.gelu = nn.GELU()
        self.pwconv1 = ConvNd[dim](C * len(kernel_sizes), C, 1)

    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs], dim=1)
        x = self.gelu(self.norm(x))
        x = self.pwconv1(x)
        x = skip + x
        return x


class ConvBlock2d(nn.Module):

    def __init__(self, c, f, block_type="convnext_like", group_divisor=1):
        super().__init__()
        if block_type == "convnext_like":
            self.conv_block = ConvNeXtLikeBlock(
                c,
                dim=2,
                kernel_sizes=[(3, 3)],
                group_divisor=group_divisor,
                padding="same",
            )
        elif block_type == "basic_resnet":
            self.conv_block = ResBasicBlock(
                c,
                c,
                f,
                stride=1,
                se_channels=min(64, max(c, 32)),
                group_divisor=group_divisor,
                use_fwSE=False,
            )
        elif block_type == "basic_resnet_fwse":
            self.conv_block = ResBasicBlock(
                c,
                c,
                f,
                stride=1,
                se_channels=min(64, max(c, 32)),
                group_divisor=group_divisor,
                use_fwSE=True,
            )
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.conv_block(x)


class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got "
                f"`embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len, bsz):
        return (tensor.view(bsz, seq_len, self.num_heads,
                            self.head_dim).transpose(1, 2).contiguous())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len,
                                   bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights,
                               p=self.dropout,
                               training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len,
                                       self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class)
        # rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        n_state,
        n_mlp,
        n_head,
        channel_last=False,
        act_do=0.0,
        att_do=0.0,
        hid_do=0.0,
        ln_eps=1e-6,
    ):

        hidden_size = n_state
        num_attention_heads = n_head
        intermediate_size = n_mlp
        activation_dropout = act_do
        attention_dropout = att_do
        hidden_dropout = hid_do
        layer_norm_eps = ln_eps

        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_dropout=activation_dropout,
            hidden_dropout=hidden_dropout,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs


class FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        activation_dropout=0.0,
        hidden_dropout=0.0,
    ):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = NewGELUActivation()
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class BasicBlock(nn.Module):
    """
    Key difference with the BasicBlock in resnet.py:
    1. If use group convolution, conv1 have same number of input/output channels
    2. No stride to downsample
    """

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        group_divisor=4,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes if group_divisor is not None else planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=in_planes //
            group_divisor if group_divisor is not None else 1,
        )

        # If using group convolution, add point-wise conv to reshape
        if group_divisor is not None:
            self.conv1pw = nn.Conv2d(in_planes, planes, 1)
        else:
            self.conv1pw = nn.Identity()

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=planes // group_divisor if group_divisor is not None else 1,
        )

        # If using group convolution, add point-wise conv to reshape
        if group_divisor is not None:
            self.conv2pw = nn.Conv2d(planes, planes, 1)
        else:
            self.conv2pw = nn.Identity()

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if planes != in_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1pw(self.conv1(x))
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2pw(self.conv2(out))
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class fwSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    link: https://arxiv.org/pdf/1709.01507.pdf
    PyTorch implementation
    """

    def __init__(self, num_freq, num_feats=64):
        super(fwSEBlock, self).__init__()
        self.squeeze = nn.Linear(num_freq, num_feats)
        self.exitation = nn.Linear(num_feats, num_freq)

        self.activation = nn.ReLU()  # Assuming ReLU, modify as needed

    def forward(self, inputs):
        # [bs, C, F, T]
        x = torch.mean(inputs, dim=[1, 3])
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)
        # Reshape and apply excitation
        x = x[:, None, :, None]
        x = inputs * x
        return x


class ResBasicBlock(nn.Module):

    def __init__(
        self,
        in_planes,
        planes,
        num_freq,
        stride=1,
        se_channels=64,
        group_divisor=4,
        use_fwSE=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes if group_divisor is not None else planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=in_planes //
            group_divisor if group_divisor is not None else 1,
        )
        if group_divisor is not None:
            self.conv1pw = nn.Conv2d(in_planes, planes, 1)
        else:
            self.conv1pw = nn.Identity()

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=planes // group_divisor if group_divisor is not None else 1,
        )

        if group_divisor is not None:
            self.conv2pw = nn.Conv2d(planes, planes, 1)
        else:
            self.conv2pw = nn.Identity()

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if use_fwSE:
            self.se = fwSEBlock(num_freq, se_channels)
        else:
            self.se = nn.Identity()

        if planes != in_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1pw(self.conv1(x))
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2pw(self.conv2(out))
        out = self.bn2(out)
        out = self.se(out)

        out += self.downsample(residual)
        out = self.relu(out)
        return out


class TimeContextBlock1d(nn.Module):
    """ """

    def __init__(
        self,
        C,
        hC,
        pos_ker_sz=59,
        block_type="att",
    ):
        super().__init__()
        assert pos_ker_sz

        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(C, hC, 1),
            LayerNorm(hC, eps=1e-6, data_format="channels_first"))

        if block_type == "fc":
            self.tcm = nn.Sequential(
                nn.Conv1d(hC, hC * 2, 1),
                LayerNorm(hC * 2, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                nn.Conv1d(hC * 2, hC, 1),
            )
        elif block_type == "gru":
            # Just GRU
            self.tcm = nn.Sequential(
                GRU(
                    input_size=hC,
                    hidden_size=hC,
                    num_layers=1,
                    bias=True,
                    batch_first=False,
                    dropout=0.0,
                    bidirectional=True,
                ),
                nn.Conv1d(2 * hC, hC, 1),
            )
        elif block_type == "att":
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                PosEncConv(hC, ks=pos_ker_sz, groups=hC),
                TransformerEncoderLayer(n_state=hC, n_mlp=hC * 2, n_head=4),
            )
        elif block_type == "conv+att":
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                ConvNeXtLikeBlock(hC,
                                  dim=1,
                                  kernel_sizes=[7],
                                  group_divisor=1,
                                  padding="same"),
                ConvNeXtLikeBlock(hC,
                                  dim=1,
                                  kernel_sizes=[19],
                                  group_divisor=1,
                                  padding="same"),
                ConvNeXtLikeBlock(hC,
                                  dim=1,
                                  kernel_sizes=[31],
                                  group_divisor=1,
                                  padding="same"),
                ConvNeXtLikeBlock(hC,
                                  dim=1,
                                  kernel_sizes=[59],
                                  group_divisor=1,
                                  padding="same"),
                TransformerEncoderLayer(n_state=hC, n_mlp=hC, n_head=4),
            )
        else:
            raise NotImplementedError()

        self.exp_dim_conv = nn.Conv1d(hC, C, 1)

    def forward(self, x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x


class ReDimNetBone(nn.Module):

    def __init__(
        self,
        F=72,
        C=16,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        stages_setup=(
            # stride, num_blocks, conv_exp, kernel_size, att_block_red
            (1, 2, 1, [(3, 3)], None),  # 16
            (2, 3, 1, [(3, 3)], None),  # 32
            # 64, (72*12 // 8) = 108 - channels in attention block
            (3, 4, 1, [(3, 3)], 8),
            (2, 5, 1, [(3, 3)], 8),  # 128
            (1, 5, 1, [(7, 1)], 8),  # 128 # TDNN - time context
            (2, 3, 1, [(3, 3)], 8),  # 256
        ),
        group_divisor=1,
        out_channels=512,
    ):
        super().__init__()
        self.F = F
        self.C = C

        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type

        self.stages_setup = stages_setup
        self.build(stages_setup, group_divisor, out_channels)

    def build(self, stages_setup, group_divisor, out_channels):
        self.num_stages = len(stages_setup)

        cur_c = self.C
        cur_f = self.F
        # Weighting the inputs
        # TODO: ask authors about the impact of this pre-weighting
        self.inputs_weights = torch.nn.ParameterList(
            [nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=False)] + [
                nn.Parameter(
                    torch.zeros(1, num_inputs + 1, self.C * self.F, 1),
                    requires_grad=True,
                ) for num_inputs in range(1,
                                          len(stages_setup) + 1)
            ])

        self.stem = nn.Sequential(
            nn.Conv2d(1, int(cur_c), kernel_size=3, stride=1, padding="same"),
            LayerNorm(int(cur_c), eps=1e-6, data_format="channels_first"),
        )

        Block1d = functools.partial(TimeContextBlock1d,
                                    block_type=self.block_1d_type)
        Block2d = functools.partial(ConvBlock2d, block_type=self.block_2d_type)

        self.stages_cfs = []
        for stage_ind, (
                stride,
                num_blocks,
                conv_exp,
                kernel_sizes,  # TODO: Why the kernel_sizes are not used?
                att_block_red,
        ) in enumerate(stages_setup):
            assert stride in [1, 2, 3]
            # Pool frequencies & expand channels if needed
            layers = [
                nn.Conv2d(
                    int(cur_c),
                    int(stride * cur_c * conv_exp),
                    kernel_size=(stride, 1),
                    stride=(stride, 1),
                    padding=0,
                    groups=1,
                ),
            ]

            self.stages_cfs.append((cur_c, cur_f))

            cur_c = stride * cur_c
            assert cur_f % stride == 0
            cur_f = cur_f // stride

            for _ in range(num_blocks):
                # ConvBlock2d(f, c, block_type="convnext_like", group_divisor=1)
                layers.append(
                    Block2d(c=int(cur_c * conv_exp),
                            f=cur_f,
                            group_divisor=group_divisor))

            if conv_exp != 1:
                # Squeeze back channels to align with ReDimNet c+f reshaping:
                _group_divisor = group_divisor
                # if c // group_divisor == 0:
                # _group_divisor = c
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            int(cur_c * conv_exp),
                            cur_c,
                            kernel_size=(3, 3),
                            stride=1,
                            padding="same",
                            groups=(cur_c // _group_divisor
                                    if _group_divisor is not None else 1),
                        ),
                        nn.BatchNorm2d(
                            cur_c,
                            eps=1e-6,
                        ),
                        nn.GELU(),
                        nn.Conv2d(cur_c, cur_c, 1),
                    ))

            layers.append(to1d())

            # reduce block?
            if att_block_red is not None:
                layers.append(
                    Block1d(self.C * self.F,
                            hC=(self.C * self.F) // att_block_red))

            setattr(self, f"stage{stage_ind}", nn.Sequential(*layers))

        if out_channels is not None:
            self.mfa = nn.Sequential(
                nn.Conv1d(self.F * self.C,
                          out_channels,
                          kernel_size=1,
                          padding="same"),
                nn.BatchNorm1d(out_channels, affine=True),
            )
        else:
            self.mfa = nn.Identity()

    def to1d(self, x):
        size = x.size()
        bs, c, f, t = tuple(size)
        return x.permute((0, 2, 1, 3)).reshape((bs, c * f, t))

    def to2d(self, x, c, f):
        size = x.size()
        bs, cf, t = tuple(size)
        return x.reshape((bs, f, c, t)).permute((0, 2, 1, 3))

    def weigth1d(self, outs_1d, i):
        xs = torch.cat([t.unsqueeze(1) for t in outs_1d], dim=1)
        w = F.softmax(self.inputs_weights[i], dim=1)
        x = (w * xs).sum(dim=1)
        return x

    def run_stage(self, prev_outs_1d, stage_ind):
        stage = getattr(self, f"stage{stage_ind}")
        c, f = self.stages_cfs[stage_ind]

        x = self.weigth1d(prev_outs_1d, stage_ind)
        x = self.to2d(x, c, f)
        x = stage(x)
        return x

    def forward(self, inp):
        x = self.stem(inp)
        outputs_1d = [self.to1d(x)]
        for stage_ind in range(self.num_stages):
            outputs_1d.append(self.run_stage(outputs_1d, stage_ind))
        x = self.weigth1d(outputs_1d, -1)
        x = self.mfa(x)
        return x


class ReDimNet(nn.Module):

    def __init__(
        self,
        feat_dim=72,
        C=16,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        # Default setup: M version:
        stages_setup=(
            # stride, num_blocks, kernel_sizes, layer_ext, att_block_red
            (1, 2, 1, [(3, 3)], 12),
            (2, 2, 1, [(3, 3)], 12),
            (1, 3, 1, [(3, 3)], 12),
            (2, 4, 1, [(3, 3)], 8),
            (1, 4, 1, [(3, 3)], 8),
            (2, 4, 1, [(3, 3)], 4),
        ),
        group_divisor=4,
        out_channels=None,
        # -------------------------
        embed_dim=192,
        pooling_func="ASTP",
        global_context_att=True,
        two_emb_layer=False,
    ):

        super().__init__()
        self.two_emb_layer = two_emb_layer
        self.backbone = ReDimNetBone(
            feat_dim,
            C,
            block_1d_type,
            block_2d_type,
            stages_setup,
            group_divisor,
            out_channels,
        )

        if out_channels is None:
            out_channels = C * feat_dim

        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att)

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
        x = x.permute(0, 2, 1)  # (B,F,T) => (B,T,F)
        x = x.unsqueeze_(1)
        out = self.backbone(x)

        return out

    def get_frame_level_feat(self, x):
        # for outer interface
        out = self._get_frame_level_feat(x).permute(0, 2, 1)

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


def ReDimNetB0(feat_dim=60,
               embed_dim=192,
               pooling_func="ASTP",
               two_emb_layer=False):
    return ReDimNet(
        feat_dim=feat_dim,
        C=10,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        stages_setup=[
            (1, 2, 1, [(3, 3)], 30),
            (2, 3, 2, [(3, 3)], 30),
            (1, 3, 3, [(3, 3)], 30),
            (2, 4, 2, [(3, 3)], 10),
            (1, 3, 1, [(3, 3)], 10),
        ],
        group_divisor=1,
        out_channels=None,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        two_emb_layer=two_emb_layer,
    )


def ReDimNetB1(feat_dim=72,
               embed_dim=192,
               pooling_func="ASTP",
               two_emb_layer=False):
    return ReDimNet(
        feat_dim=feat_dim,
        C=12,
        block_1d_type="conv+att",
        block_2d_type="convnext_like",
        stages_setup=[
            (1, 2, 1, [(3, 3)], None),
            (2, 3, 1, [(3, 3)], None),
            (3, 4, 1, [(3, 3)], 12),
            (2, 5, 1, [(3, 3)], 12),
            (2, 3, 1, [(3, 3)], 8),
        ],
        group_divisor=8,
        out_channels=None,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        two_emb_layer=two_emb_layer,
    )


def ReDimNetB2(feat_dim=72,
               embed_dim=192,
               pooling_func="ASTP",
               two_emb_layer=False):
    return ReDimNet(
        feat_dim=feat_dim,
        C=16,
        block_1d_type="conv+att",
        block_2d_type="convnext_like",
        stages_setup=[
            (1, 2, 1, [(3, 3)], 12),
            (2, 2, 1, [(3, 3)], 12),
            (1, 3, 1, [(3, 3)], 12),
            (2, 4, 1, [(3, 3)], 8),
            (1, 4, 1, [(3, 3)], 8),
            (2, 4, 1, [(3, 3)], 4),
        ],
        group_divisor=4,
        out_channels=None,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        two_emb_layer=two_emb_layer,
    )


def ReDimNetB3(feat_dim=72,
               embed_dim=192,
               pooling_func="ASTP",
               two_emb_layer=False):
    return ReDimNet(
        feat_dim=feat_dim,
        C=16,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet_fwse",
        stages_setup=[
            (1, 6, 4, [(3, 3)], 32),
            (2, 6, 2, [(3, 3)], 32),
            (1, 8, 2, [(3, 3)], 32),
            (2, 10, 2, [(3, 3)], 16),
            (1, 10, 1, [(3, 3)], 16),
            (2, 8, 1, [(3, 3)], 16),
        ],
        group_divisor=1,
        out_channels=None,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        two_emb_layer=two_emb_layer,
    )


def ReDimNetB4(feat_dim=72,
               embed_dim=192,
               pooling_func="ASTP",
               two_emb_layer=False):
    return ReDimNet(
        feat_dim=feat_dim,
        C=32,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet_fwse",
        stages_setup=[
            (1, 4, 2, [(3, 3)], 48),
            (2, 4, 2, [(3, 3)], 48),
            (1, 6, 2, [(3, 3)], 48),
            (2, 6, 1, [(3, 3)], 32),
            (1, 8, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], 16),
        ],
        group_divisor=1,
        out_channels=None,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        two_emb_layer=two_emb_layer,
    )


def ReDimNetB5(feat_dim=72,
               embed_dim=192,
               pooling_func="ASTP",
               two_emb_layer=False):
    return ReDimNet(
        feat_dim=feat_dim,
        C=32,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet_fwse",
        stages_setup=[
            (1, 4, 2, [(3, 3)], 48),
            (2, 4, 2, [(3, 3)], 48),
            (1, 6, 2, [(3, 3)], 48),
            (2, 6, 1, [(3, 3)], 32),
            (1, 8, 1, [(3, 3)], 24),
            (2, 4, 1, [(3, 3)], 16),
        ],
        group_divisor=16,
        out_channels=None,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        two_emb_layer=two_emb_layer,
    )


def ReDimNetB6(feat_dim=72,
               embed_dim=192,
               pooling_func="ASTP",
               two_emb_layer=False):
    return ReDimNet(
        feat_dim=feat_dim,
        C=32,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        stages_setup=[
            (1, 4, 4, [(3, 3)], 32),
            (2, 6, 2, [(3, 3)], 32),
            (1, 6, 2, [(3, 3)], 24),
            (3, 8, 1, [(3, 3)], 24),
            (1, 8, 1, [(3, 3)], 16),
            (2, 8, 1, [(3, 3)], 16),
        ],
        group_divisor=32,
        out_channels=None,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        two_emb_layer=two_emb_layer,
    )


if __name__ == "__main__":
    x = torch.zeros(1, 200, 72)
    model = ReDimNet(feat_dim=72, embed_dim=192, two_emb_layer=False)
    model.eval()
    out = model(x)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))

    # Currently, the model sizes differ from the ones in the paper
    model_classes = [
        ReDimNetB0,  # 1.0M v.s. 1.0M
        ReDimNetB1,  # 2.1M v.s. 2.2M
        ReDimNetB2,  # 4.9M v.s. 4.7M
        ReDimNetB3,  # 3.2M v.s. 3.0M
        ReDimNetB4,  # 6.4M v.s. 6.3M
        ReDimNetB5,  # 7.65M v.s. 9.2M
        ReDimNetB6,  # 15.0M v.s. 15.0M
    ]

    for i, model_class in enumerate(model_classes):
        model = model_class()
        num_params = sum(p.numel() for p in model.parameters())
        print("{} M of Model B{}".format(num_params / 1e6, i))
