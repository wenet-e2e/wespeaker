# Copyright (c) 2024 Yiyang Zhao (zhaoyy22@mails.tsinghua.edu.cn)
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


import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch import nn

from typing import Iterable, Optional

import os
import hashlib
import whisper
import logging
import urllib.request


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(
                x.dtype), None if self.bias is None else self.bias.to(
                x.dtype))


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(
        length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None),
            # will prepend the cached kv tensors; otherwise,
            # perform key/value projections for self- or
            # cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once
            # and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int,
                 cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(
            n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(
                n_state, n_mlp), nn.GELU(), Linear(
                n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x),
                                    xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
            self,
            n_mels: int,
            n_ctx: int,
            n_state: int,
            n_head: int,
            n_layer: int,
            layer_st: int,
            layer_ed: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(
            n_state,
            n_state,
            kernel_size=3,
            stride=2,
            padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        # self.ln_post = LayerNorm(n_state)
        # ------------------------ADD:add new layer norm------------------------
        self.ln_post2 = LayerNorm(n_state * (layer_ed - layer_st + 1))

        self.layer_st = layer_st
        self.layer_ed = layer_ed

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # ---------------------------ADD------------------------
        x = x.permute(0, 2, 1)

        x = x.squeeze(1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # ------------Change:Tailor the positional_embedding----------
        assert x.shape[2:] == self.positional_embedding.shape[1:], \
            "incorrect audio shape"
        if self.positional_embedding.shape[0] > x.shape[1]:
            temp_positional_embedding = self.positional_embedding[:x.shape[1], :]
        elif self.positional_embedding.shape[0] < x.shape[1]:
            x = x[:, :self.positional_embedding.shape[0], :]
            temp_positional_embedding = self.positional_embedding
        else:
            temp_positional_embedding = self.positional_embedding

        x = (x + temp_positional_embedding).to(x.dtype)

        # ----------Change: Concat block outputs------
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.layer_st <= i <= self.layer_ed:
                out.append(x)

        xs = torch.cat(out, dim=-1)

        xs = self.ln_post2(xs)
        return xs


class whisper_encoder(torch.nn.Module):
    def __init__(self,
                 frozen=False,
                 n_mels=80,
                 num_blocks=24,
                 output_size=1280,
                 n_head=20,
                 layer_st=16,
                 layer_ed=23,
                 model_path=None,
                 sample_rate=16000
                 ):
        super(whisper_encoder, self).__init__()
        self.encoder = AudioEncoder(
            n_mels=n_mels,
            n_layer=num_blocks,
            n_state=output_size,
            n_ctx=1500,
            n_head=n_head,
            layer_st=layer_st,
            layer_ed=layer_ed)
        # 0 for freeze finetune, 1 for all parameters finetune
        self.frozen = frozen
        self.single_output_size = output_size
        self.concat_layer = layer_ed - layer_st + 1
        self.n_mels = n_mels

        # load model
        if model_path:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    self._download_whisper_model(model_path)
                dist.barrier()  # Wait for rank 0 to finish downloading
                self._load_pretrained_weights(model_path)
            else:
                self._download_whisper_model(model_path)
                self._load_pretrained_weights(model_path)

        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    def _download_whisper_model(self, model_path='whisper_hub/large-v2.pt'):
        download_dir = os.path.dirname(model_path)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        if not os.path.isfile(model_path):
            print("Downloading large-v2.pt ...")
            url = 'https://openaipublic.azureedge.net/main/whisper/models/' \
                '81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/' \
                'large-v2.pt'

            urllib.request.urlretrieve(url, model_path)

            md5 = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

            if md5 != "668764447eeda98eeba5ef7bfcb4cc3d":
                print("Wrong md5sum of large-v2.pt")
                os.remove(model_path)
                raise ValueError("MD5 checksum does not match!")
        else:
            print("Model already downloaded.")

    def _load_pretrained_weights(self, model_path):
        print(f"Loading pretrained weights from {model_path}...")

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = state_dict['model_state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('encoder.', '', 1)
            new_state_dict[new_key] = v

        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            new_state_dict, strict=False)
        print("Pretrained weights loaded successfully.")
        for key in missing_keys:
            logging.warning('missing tensor: {}'.format(key))
        for key in unexpected_keys:
            logging.warning('unexpected tensor: {}'.format(key))

    def output_size(self):
        return self.single_output_size * self.concat_layer

    def forward(self, wavs, wavs_len):
        with torch.no_grad():
            processed_feats = []
            for i in range(wavs.size(0)):
                tf_tensor = wavs[i].unsqueeze(0).to(wavs.device)
                mat = whisper.log_mel_spectrogram(
                    tf_tensor.squeeze(), n_mels=self.n_mels)
                processed_feats.append(mat)

            feat = torch.stack(processed_feats, dim=0).to(wavs.device)

        feat = feat.transpose(1, 2)
        # (B,T,F)
        x = self.encoder(feat)
        return x, None
