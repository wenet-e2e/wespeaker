# Copyright (c) 2026 Bosen Xu (2332974001@qq.com)
# Based on original code from PalabraAI/redimnet2
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
import wespeaker.models.pooling_layers as pooling_layers
from wespeaker.frontend.tfmel import (TFMelBanks, TFSpectrogram,
                                      NormalizeAudio, PreEmphasis, FbankAug)


def ShapeLogger(x):
    return x


class FreqEncoder(nn.Module):
    def __init__(self, c, bins):
        super().__init__()
        self.freq_embedder = nn.Embedding(
            num_embeddings=bins,
            embedding_dim=c)

    def forward(self, x):
        b, c, f, t = x.size()
        freqs = torch.range(start=0, end=f - 1, step=1, dtype=torch.long)
        freqs = freqs.unsqueeze(0).repeat(b, 1).to(x.device)  # [bs,f]
        fe = self.freq_embedder(freqs).permute(
            0, 2, 1).unsqueeze(-1)  # [bs, freq_emb_dim, f, 1]
        fe = fe.repeat(1, 1, 1, t)
        x = x + fe
        return x


class LayerNorm(nn.Module):
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


class fwSEBlock(nn.Module):
    def __init__(self, num_freq, num_feats=64):
        super(fwSEBlock, self).__init__()
        self.squeeze = nn.Linear(num_freq, num_feats)
        self.exitation = nn.Linear(num_feats, num_freq)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = torch.mean(inputs, dim=[1, 3])
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)
        x = x[:, None, :, None]
        x = inputs * x
        return x


class ResBasicBlock(nn.Module):
    def __init__(self, inc, outc, num_freq, stride=1, se_channels=64,
                 Gdiv=4, use_fwSE=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inc,
            inc if Gdiv is not None else outc,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=inc //
            Gdiv if Gdiv is not None else 1)
        if Gdiv is not None:
            self.conv1pw = nn.Conv2d(inc, outc, 1)
        else:
            self.conv1pw = nn.Identity()
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False,
                               groups=outc // Gdiv if Gdiv is not None else 1)
        if Gdiv is not None:
            self.conv2pw = nn.Conv2d(outc, outc, 1)
        else:
            self.conv2pw = nn.Identity()
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
        if use_fwSE:
            self.se = fwSEBlock(num_freq, se_channels)
        else:
            self.se = nn.Identity()
        if outc != inc:
            self.downsample = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outc),
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


ConvNd = {1: nn.Conv1d, 2: nn.Conv2d}
BatchNormNd = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d}


class ConvNeXtLikeBlock(nn.Module):
    def __init__(self, C, dim=2, kernel_sizes=None, Gdiv=1, padding='same',
                 activation='gelu'):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [(3, 3)]
        self.dwconvs = nn.ModuleList(
            modules=[
                ConvNd[dim](
                    C,
                    C,
                    kernel_size=ks,
                    padding=padding,
                    groups=C //
                    Gdiv if Gdiv is not None else 1) for ks in kernel_sizes])
        self.norm = BatchNormNd[dim](C * len(kernel_sizes))
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        self.pwconv1 = ConvNd[dim](C * len(kernel_sizes), C, 1)

    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs], dim=1)
        x = self.act(self.norm(x))
        x = self.pwconv1(x)
        x = skip + x
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(
            bsz, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class NewGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        self.act = F.gelu if not use_gelu_python else self._gelu_python

    def _gelu_python(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(input)


class FastGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(
            input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_fast": FastGELUActivation,
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}
ACT2FN = {k: v() if isinstance(v, type) else v for k, v in ACT2CLS.items()}


class FeedForward(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str = 'gelu_new',
            activation_dropout: float = 0.0,
            hidden_dropout: float = 0.0):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN.get(hidden_act, GELUActivation())
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            n_state: int,
            n_mlp: int,
            n_head: int,
            channel_last: bool = False,
            act: str = 'gelu_new',
            act_do: float = 0.0,
            att_do: float = 0.0,
            hid_do: float = 0.0,
            ln_eps: float = 1e-6):
        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(
            embed_dim=n_state, num_heads=n_head, dropout=att_do)
        self.layer_norm = nn.LayerNorm(n_state, eps=ln_eps)
        self.feed_forward = FeedForward(
            hidden_size=n_state,
            hidden_act=act,
            intermediate_size=n_mlp,
            activation_dropout=act_do,
            hidden_dropout=hid_do)
        self.final_layer_norm = nn.LayerNorm(n_state, eps=ln_eps)

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


class MelBanks(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        f_min=20,
        f_max=7600,
        n_mels=80,
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        spec_norm='mn',
        freq_start_bin=0,
        num_apply_spec_aug=1,
        freq_mask_width=(
            0,
            8),
        time_mask_width=(
            0,
            10)):
        super(MelBanks, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        import torchaudio
        self.torchfbank = torch.nn.Sequential(
            NormalizeAudio() if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                window_fn=torch.hamming_window),
        )
        self.spec_norm = spec_norm
        if spec_norm == 'mn':
            self.spec_norm = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
        elif spec_norm == 'mvn':
            self.spec_norm = lambda x: (x - torch.mean(x, dim=-1, keepdims=True)) / \
                (torch.std(x, dim=-1, keepdim=True) + 1e-8)
        elif spec_norm == 'bn':
            self.spec_norm = nn.BatchNorm1d(n_mels)
        else:
            self.spec_norm = lambda x: x
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width)
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = self.spec_norm(x)
            if self.training:
                for _ in range(self.num_apply_spec_aug):
                    x = self.specaug(x)
        return x.to(xdtype)


class to1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        bs, c, f, t = size
        return x.permute((0, 2, 1, 3)).reshape((bs, c * f, t))


class to2d(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self, x):
        bs, cf, t = x.size()
        out = x.reshape((bs, self.f, self.c, t)).permute((0, 2, 1, 3))
        return out


class weigth1d(nn.Module):
    def __init__(self, N, C, sequential=False, requires_grad=True):
        super().__init__()
        self.N = N
        self.sequential = sequential
        self.w = nn.Parameter(torch.zeros(1, N, C, 1), requires_grad=requires_grad)

    def forward(self, xs):
        w = F.softmax(self.w, dim=1)
        if not self.sequential:
            xs = torch.cat([t.unsqueeze(1) for t in xs], dim=1)
            x = (w * xs).sum(dim=1)
        else:
            s = torch.zeros_like(xs[0])
            for i, t in enumerate(xs):
                s += t * w[:, i, :, :]
            x = s
        return x


class ConvBlock2d(nn.Module):
    def __init__(self, c, f, block_type="convnext_like", Gdiv=1, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [(3, 3)]
        if block_type == "convnext_like":
            self.conv_block = ConvNeXtLikeBlock(
                c, dim=2, kernel_sizes=kernel_sizes, Gdiv=Gdiv,
                padding='same', activation='gelu')
        elif block_type == "convnext_like_relu":
            self.conv_block = ConvNeXtLikeBlock(
                c, dim=2, kernel_sizes=kernel_sizes, Gdiv=Gdiv,
                padding='same', activation='relu')
        elif block_type == "basic_resnet":
            self.conv_block = ResBasicBlock(
                c, c, f, stride=1, se_channels=min(
                    64, max(
                        c, 32)), Gdiv=Gdiv, use_fwSE=False)
        elif block_type == "basic_resnet_fwse":
            self.conv_block = ResBasicBlock(
                c, c, f, stride=1, se_channels=min(
                    64, max(
                        c, 32)), Gdiv=Gdiv, use_fwSE=True)
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.conv_block(x)


class PosEncConv(nn.Module):
    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(
            C, C, ks, padding=ks // 2,
            groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return x + self.norm(self.conv(x))


class TimeContextBlock1d(nn.Module):
    def __init__(self, C, hC, pos_ker_sz=59, block_type='att',
                 red_dim_conv=None, exp_dim_conv=None):
        super().__init__()
        assert pos_ker_sz
        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(C, hC, 1),
            LayerNorm(hC, eps=1e-6, data_format="channels_first")
        )
        if block_type == 'fc':
            self.tcm = nn.Sequential(
                nn.Conv1d(hC, hC * 2, 1),
                LayerNorm(hC * 2, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                nn.Conv1d(hC * 2, hC, 1)
            )
        elif block_type == 'conv':
            self.tcm = nn.Sequential(
                *[ConvNeXtLikeBlock(
                    hC, dim=1, kernel_sizes=[7, 15, 31],
                    Gdiv=1, padding='same') for i in range(4)])
        elif block_type == 'att':
            self.tcm = nn.Sequential(
                PosEncConv(hC, ks=pos_ker_sz, groups=hC),
                TransformerEncoderLayer(n_state=hC, n_mlp=hC * 2, n_head=4)
            )
        elif block_type == 'conv+att':
            self.tcm = nn.Sequential(
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding='same'),
                TransformerEncoderLayer(n_state=hC, n_mlp=hC, n_head=4)
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


class ReDimNet2(nn.Module):
    # UNet-like ReDimNet
    def __init__(self,
                 F=72,
                 C=24,
                 spec_in_channels=1,  # Phase + Magnitude
                 causal='none',
                 out_channels=None,
                 block_1d_type='tf-att',
                 block_2d_type="basic_resnet",
                 return_2d_output=False,
                 fm_weigthing_type='NC',
                 use_freq_pos_enc=False,
                 compress_tconvs=True,
                 stages_setup=None,
                 group_divisor=1,
                 dual_agg=False,
                 agg_gnorm=False,
                 # Subnet stuff
                 return_all_outputs=False,
                 offset_fm_weights=0,
                 is_subnet=False,
                 ):
        super().__init__()
        if stages_setup is None:
            stages_setup = [
                # Encoder part:
                ((1, 1), 2, 4, [(3, 3)], None),  # 16
                ((2, 1), 3, 3, [(3, 3)], None),  # 32

                ((1, 2), 4, 2, [(3, 3)], None),  # 64,
                ((2, 1), 5, 1, [(3, 3)], 48),  # 128

                ((1, 2), 4, 1, [(3, 3)], 64),  # 128
                ((2, 1), 3, 1, [(3, 3)], 96),  # 128
            ]
        self.F = F
        self.C = C

        if causal == 'full':
            block_1d_type = block_1d_type + '-causal'
            block_2d_type = block_2d_type + '-causal'
            self.causal = True
        elif causal == 'only_1d':
            block_1d_type = block_1d_type + '-causal'
            self.causal = True
        elif causal == 'none':
            self.causal = False
        else:
            raise NotImplementedError()

        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type

        self.stages_setup = stages_setup
        self.fm_weigthing_type = fm_weigthing_type
        self.dual_agg = dual_agg
        self.agg_gnorm = agg_gnorm

        # Subnet stuff
        self.is_subnet = is_subnet
        self.offset_fm_weights = offset_fm_weights
        self.return_all_outputs = return_all_outputs

        self.build(F, C, spec_in_channels, out_channels, stages_setup, group_divisor,
                   compress_tconvs, return_2d_output, use_freq_pos_enc)

    def build(self, F, C, spec_in_channels, out_channels, stages_setup, group_divisor,
              compress_tconvs, return_2d_output, use_freq_pos_enc):
        self.F = F
        self.C = C

        c = C
        f = F

        stt = 1
        sft = 1

        max_stt = stt

        self.num_stages = len(stages_setup)

        append_to1d_before_tcm = True
        Block1d = functools.partial(
            TimeContextBlock1d, block_type=self.block_1d_type)
        Block2d = functools.partial(
            ConvBlock2d, block_type=self.block_2d_type)

        if self.fm_weigthing_type == 'NC':
            agg1d = functools.partial(weigth1d, C=F * C)
        elif self.fm_weigthing_type == 'N':
            agg1d = functools.partial(weigth1d, C=None)
        else:
            raise NotImplementedError()

        if not self.is_subnet:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    spec_in_channels, int(c), kernel_size=3,
                    stride=1, padding='same'),
                LayerNorm(int(c), eps=1e-6, data_format="channels_first"),
                to1d()
            )
        else:
            # Subnet stem: aggregate offset_fm_weights incoming 1D feature maps,
            # reshape to 2D, then apply a standard conv+norm stem before to1d().
            assert self.offset_fm_weights > 0, \
                "offset_fm_weights must be > 0 when is_subnet=True"
            self.stem = nn.Sequential(
                agg1d(
                    N=self.offset_fm_weights,
                    requires_grad=self.offset_fm_weights > 1),
                to2d(f=F, c=C),
                nn.Conv2d(
                    int(c), int(c), kernel_size=3,
                    stride=1, padding='same'),
                LayerNorm(int(c), eps=1e-6, data_format="channels_first"),
                to1d()
            )

        if self.agg_gnorm:
            self.stem_gnorm = nn.GroupNorm(num_groups=C, num_channels=C * F)

        # Track accumulated feature-map count for the weight1d N parameter.
        # Starts at offset_fm_weights+1 to account for the
        # subnet offset + stem output.
        feat_count = self.offset_fm_weights + 1
        self._stage_has_dual = []

        for stage_ind, (stride, num_blocks, conv_exp, kernel_sizes,
                        att_block_red) in enumerate(stages_setup):
            (sf, st) = stride
            tot_stride = np.prod((sf, st))
            num_feats_to_weight = feat_count
            # if tot_stride > 1:
            layers = []
            sft = sft * sf
            stt = stt * st
            layers.append(
                agg1d(
                    N=num_feats_to_weight,
                    requires_grad=num_feats_to_weight > 1))
            layers.append(to2d(f=f, c=c))
            if use_freq_pos_enc:
                layers.append(FreqEncoder(c=c, bins=f))

            layers.append(ShapeLogger(nn.Conv2d(
                int(c), int(sf * c * conv_exp),
                kernel_size=(sf, stt),
                stride=(sf, stt),
                padding=0,
                groups=1 if not compress_tconvs else
                math.gcd(int(c), int(sf * c * conv_exp)))))

            c = sf * c
            assert f % sf == 0
            f = f // sf

            if stt >= max_stt:
                max_stt = stt

            for block_ind in range(num_blocks):
                layers.append(
                    Block2d(c=int(c * conv_exp), f=f,
                            kernel_sizes=kernel_sizes, Gdiv=group_divisor))

            if conv_exp != 1:
                _group_divisor = group_divisor
                layers.append(nn.Sequential(
                    nn.Conv2d(
                        int(c * conv_exp), c, kernel_size=1,
                        stride=1, padding='same'),
                    nn.BatchNorm2d(c, eps=1e-6)
                ))

            has_dual = self.dual_agg and att_block_red is not None

            if has_dual:
                # Split the stage so the 1D-attention branch runs in
                # parallel with a plain 2D->1D reshape branch; both are
                # upsampled (+gnorm) and aggregated alongside prior
                # feature maps.
                if append_to1d_before_tcm:
                    layers.append(to1d())
                setattr(self, f'stage{stage_ind}_pre', nn.Sequential(*layers))

                blk_1d = Block1d(C * F, hC=(C * F) // att_block_red)
                setattr(self, f'stage{stage_ind}_1d', blk_1d)

                up_2d = [ShapeLogger(nn.Upsample(scale_factor=stt, mode='nearest'))]
                if self.agg_gnorm:
                    up_2d.append(nn.GroupNorm(num_groups=C, num_channels=C * F))
                setattr(self, f'stage{stage_ind}_up_2d', nn.Sequential(*up_2d))

                up_1d = [ShapeLogger(nn.Upsample(scale_factor=stt, mode='nearest'))]
                if self.agg_gnorm:
                    up_1d.append(nn.GroupNorm(num_groups=C, num_channels=C * F))
                setattr(self, f'stage{stage_ind}_up_1d', nn.Sequential(*up_1d))

                self._stage_has_dual.append(True)
                feat_count += 2
            else:
                if append_to1d_before_tcm:
                    layers.append(to1d())
                if att_block_red is not None:
                    if append_to1d_before_tcm:
                        layers.append(Block1d(C * F, hC=(C * F) // att_block_red))
                    else:
                        layers.append(Block1d(C=c, F=f, hC=att_block_red))
                if not append_to1d_before_tcm:
                    layers.append(to1d())
                layers.append(
                    ShapeLogger(nn.Upsample(scale_factor=stt, mode='nearest')))
                if self.agg_gnorm:
                    layers.append(nn.GroupNorm(num_groups=C, num_channels=C * F))
                setattr(self, f'stage{stage_ind}', nn.Sequential(*layers))

                self._stage_has_dual.append(False)
                feat_count += 1

        self.fin_wght1d = agg1d(N=feat_count, requires_grad=feat_count > 1)

        self.time_stride = max_stt
        self.freq_stride = sft
        self.head = nn.Identity()
        print(f"out_channels : {out_channels}")
        if return_2d_output:
            self.fin_to2d = to2d(f=f, c=c)
            if out_channels is not None:
                self.head = nn.Conv2d(c, out_channels, 1)
        else:
            self.fin_to2d = nn.Identity()
            if out_channels is not None:
                self.head = nn.Conv1d(C * F, out_channels, 1)

    def run_stage(self, prev_outs_1d, stage_ind):
        if self._stage_has_dual[stage_ind]:
            pre = getattr(self, f'stage{stage_ind}_pre')
            blk_1d = getattr(self, f'stage{stage_ind}_1d')
            up_2d = getattr(self, f'stage{stage_ind}_up_2d')
            up_1d = getattr(self, f'stage{stage_ind}_up_1d')
            x_pre = pre(prev_outs_1d)
            x_2d = up_2d(x_pre)
            x_1d = up_1d(blk_1d(x_pre))
            return [x_2d, x_1d]
        stage = getattr(self, f'stage{stage_ind}')
        return [stage(prev_outs_1d)]

    def forward(self, inp):
        if not self.is_subnet:
            bs, _, _, T = inp.size()
            inp = inp[:, :, :, :(T // self.time_stride) * self.time_stride]
            # Needed for right reshape operations
            x = self.stem(inp)
            if self.agg_gnorm:
                x = self.stem_gnorm(x)
            outputs_1d = [x]
        else:
            assert isinstance(inp, list), \
                "Subnet-mode ReDimNet2 expects a list of 1D feature maps as input"
            outputs_1d = list(inp)
            x = self.stem(inp)
            if self.agg_gnorm:
                x = self.stem_gnorm(x)
            outputs_1d.append(x)

        for stage_ind in range(self.num_stages):
            outputs_1d.extend(self.run_stage(outputs_1d, stage_ind))
        x = self.fin_wght1d(outputs_1d)
        outputs_1d.append(x)
        x = self.fin_to2d(x)
        x = self.head(x)

        if self.return_all_outputs:
            return x, outputs_1d
        return x


class ReDimNet2Wrap(nn.Module):
    def __init__(self,
                 F=72,
                 C=24,
                 feat_dim=None,
                 embed_dim=192,
                 pooling_func="ASTP",
                 two_emb_layer=False,
                 causal='none',
                 spec='fbank',
                 spec_in_channels=1,  # Phase + Magnitude
                 out_channels=None,
                 block_1d_type='conv+att',
                 block_2d_type="basic_resnet",
                 compress_tconvs=True,
                 return_2d_output=False,
                 use_freq_pos_enc=False,
                 fm_weigthing_type='NC',
                 stages_setup=None,
                 group_divisor=1,
                 dual_agg=False,
                 agg_gnorm=False,
                 num_classes=None,
                 feat_agg_dropout=0.0,
                 head_activation=None,
                 hop_length=160,
                 pad_right_samples=None,
                 before_pool_offset=None,
                 feat_type='pt',
                 global_context_att=True,
                 emb_bn=False,
                 spec_params=None,
                 return_all_outputs=False,
                 ):
        super().__init__()

        if stages_setup is None:
            stages_setup = [
                # Encoder part:
                ((1, 1), 2, 4, [(3, 3)], 24),  # 16
                ((2, 1), 3, 3, [(3, 3)], 24),  # 32

                ((1, 2), 4, 2, [(3, 3)], 24),  # 64,
                ((2, 1), 5, 1, [(3, 3)], 24),  # 128

                ((1, 2), 4, 1, [(3, 3)], 24),  # 128
                ((2, 1), 3, 1, [(3, 3)], 24),  # 128
            ]

        if spec_params is None:
            spec_params = dict(
                do_spec_aug=False,
                freq_mask_width=(0, 6),
                time_mask_width=(0, 8),
            )

        if feat_dim is not None:
            F = feat_dim

        self.return_all_outputs = return_all_outputs

        self.backbone = ReDimNet2(
            F=F, C=C,
            causal=causal,
            spec_in_channels=spec_in_channels,  # Phase + Magnitude
            out_channels=out_channels,
            return_2d_output=return_2d_output,
            block_1d_type=block_1d_type,
            block_2d_type=block_2d_type,
            compress_tconvs=compress_tconvs,
            fm_weigthing_type=fm_weigthing_type,
            use_freq_pos_enc=use_freq_pos_enc,
            stages_setup=stages_setup,
            group_divisor=group_divisor,
            dual_agg=dual_agg,
            agg_gnorm=agg_gnorm,
            return_all_outputs=return_all_outputs,
        )
        if spec is None or spec == 'fbank':
            self.spec = None
        elif spec == 'pt':
            self.spec = MelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif spec == 'tf':
            self.spec = TFMelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif spec == 'tf_spec':
            self.spec = TFSpectrogram(**spec_params)
        elif spec == 'pt_stft':
            self.spec = None  # STFT not implemented

        if out_channels is None:
            out_channels = C * F
        else:
            if return_2d_output:
                out_channels = (F // self.backbone.freq_stride) * out_channels
            else:
                out_channels = out_channels

        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att)

        self.pad_right_samples = pad_right_samples
        self.before_pool_offset = before_pool_offset
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.embed_dim = embed_dim
        self.emb_bn = emb_bn
        if emb_bn:  # better in SSL for SV
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = None

    def forward(self, x):
        if self.pad_right_samples is not None:
            x = torch.nn.functional.pad(
                x, (0, self.pad_right_samples),
                mode='constant', value=None)
        if self.spec is not None and self.spec != 'fbank':
            x = self.spec(x)

        if x.ndim == 3:
            x = x.unsqueeze(1)
        if self.return_all_outputs:
            out, all_outs_1d = self.backbone(x)
        else:
            out = self.backbone(x)
        # print(f"pre pool : {out.size()}")
        if out.ndim == 4:
            bs, C, F, T = out.size()
            out = out.reshape(bs, C * F, T)
        if self.before_pool_offset is not None:
            out = out[:, :, self.before_pool_offset:]
        out = self.bn(self.pool(out))
        out = self.linear(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.return_all_outputs:
            return out, all_outs_1d
        return out

    def prepare_for_frontend(self, frontend_type):
        if frontend_type == 'tfmel' and self.spec is not None and \
           self.spec != 'fbank':
            print(
                f"ReDimNet2Wrap: Disabling internal spec ({self.spec}) "
                f"for external {frontend_type} frontend")
            self.spec = None


def ReDimNet2Custom(**kwargs):
    return ReDimNet2Wrap(**kwargs)


def ReDimNet2B0(C=12, out_channels=64, stages_setup=None, **kwargs):
    if stages_setup is None:
        stages_setup = [
            [[1, 1], 2, 2, [[3, 3]], 36],
            [[2, 1], 3, 1, [[3, 3]], 36],
            [[1, 2], 4, 1, [[3, 3]], 36],
            [[2, 1], 5, 1, [[3, 3]], 36],
            [[1, 2], 4, 1, [[3, 3]], 18],
            [[2, 1], 3, 1, [[3, 3]], 18],
        ]
    return ReDimNet2Wrap(
        C=C,
        out_channels=out_channels,
        stages_setup=stages_setup,
        **kwargs,
    )


def ReDimNet2B1(C=16, out_channels=64, stages_setup=None, **kwargs):
    if stages_setup is None:
        stages_setup = [
            [[1, 1], 2, 2, [[3, 3]], 32],
            [[2, 1], 3, 1, [[3, 3]], 32],
            [[1, 2], 4, 1, [[3, 3]], 32],
            [[2, 1], 5, 1, [[3, 3]], 32],
            [[1, 2], 4, 1, [[3, 3]], 16],
            [[2, 1], 3, 1, [[3, 3]], 16],
        ]
    return ReDimNet2Wrap(
        C=C,
        out_channels=out_channels,
        stages_setup=stages_setup,
        **kwargs,
    )


def ReDimNet2B2(C=20, out_channels=64, stages_setup=None, **kwargs):
    if stages_setup is None:
        stages_setup = [
            [[1, 1], 2, 2, [[3, 5]], 40],
            [[2, 1], 3, 1, [[3, 5]], 30],
            [[1, 2], 4, 1, [[3, 5]], 30],
            [[3, 1], 5, 1, [[3, 5]], 20],
            [[1, 2], 4, 1, [[3, 7]], 20],
            [[2, 1], 3, 1, [[3, 7]], 10],
        ]
    return ReDimNet2Wrap(
        C=C,
        out_channels=out_channels,
        stages_setup=stages_setup,
        **kwargs,
    )


def ReDimNet2B3(C=24, out_channels=64, stages_setup=None, **kwargs):
    if stages_setup is None:
        stages_setup = [
            [[1, 1], 2, 2, [[3, 3]], 36],
            [[2, 1], 3, 1, [[3, 3]], 36],
            [[1, 2], 4, 1, [[3, 3]], 36],
            [[2, 1], 5, 1, [[3, 3]], 36],
            [[1, 2], 4, 1, [[3, 3]], 18],
            [[2, 1], 3, 1, [[3, 3]], 18],
        ]
    return ReDimNet2Wrap(
        C=C,
        out_channels=out_channels,
        stages_setup=stages_setup,
        **kwargs,
    )


def ReDimNet2B4(C=32, stages_setup=None, **kwargs):
    if stages_setup is None:
        stages_setup = [
            [[1, 1], 2, 4, [[3, 3]], 24],
            [[2, 1], 3, 3, [[3, 3]], 24],
            [[1, 2], 4, 2, [[3, 3]], 24],
            [[2, 1], 5, 1, [[3, 3]], 24],
            [[1, 2], 4, 1, [[3, 3]], 24],
            [[2, 1], 3, 1, [[3, 3]], 24],
        ]
    return ReDimNet2Wrap(
        C=C,
        stages_setup=stages_setup,
        **kwargs,
    )


def ReDimNet2B5(C=48, out_channels=256, stages_setup=None, **kwargs):
    if stages_setup is None:
        stages_setup = [
            [[1, 1], 2, 4, [[3, 3]], 48],
            [[2, 1], 3, 3, [[3, 3]], 48],
            [[1, 2], 4, 2, [[3, 3]], 48],
            [[2, 1], 5, 1, [[3, 3]], 48],
            [[1, 2], 4, 1, [[3, 3]], 32],
            [[2, 1], 3, 1, [[3, 3]], 32],
        ]
    return ReDimNet2Wrap(
        C=C,
        out_channels=out_channels,
        stages_setup=stages_setup,
        **kwargs,
    )


def ReDimNet2B6(C=64, out_channels=224, return_2d_output=True,
                stages_setup=None, **kwargs):
    if stages_setup is None:
        stages_setup = [
            [[1, 1], 3, 3, [[3, 3]], 64],
            [[2, 1], 4, 2, [[3, 3]], 64],
            [[1, 2], 5, 2, [[3, 3]], 48],
            [[2, 1], 5, 1, [[3, 3]], 48],
            [[1, 2], 4, 0.75, [[3, 3]], 32],
            [[2, 1], 3, 0.5, [[3, 3]], 24],
        ]
    return ReDimNet2Wrap(
        C=C,
        out_channels=out_channels,
        return_2d_output=return_2d_output,
        stages_setup=stages_setup,
        **kwargs,
    )
