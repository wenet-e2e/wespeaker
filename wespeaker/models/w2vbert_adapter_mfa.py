# Copyright (c) 2025 Qituan Shangguan (2369144677@qq.com)
# Based on original code from deeplab/3D-Speaker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import torch.nn as nn
import wespeaker.models.pooling_layers as pooling_layers


class W2VBert_Adapter_MFA(nn.Module):
    """
    Speaker Model using W2VBert Frontend and Adapter MFA backend.
    """

    def __init__(
        self,
        feat_dim: int,
        embed_dim: int = 256,
        pooling_func: str = "ASP",
        n_mfa_layers: int = -1,
        adapter_dim: int = 128,
        dropout: float = 0.0,
        num_frontend_hidden_layers: int = 24,
    ):
        """
        Args:
            feat_dim (int): Hidden dim D of each input hidden state
                (B, T, D). Should match frontend output_size().
            embed_dim (int): Final speaker embedding dim.
            pooling_func (str): Pooling layer name.
            n_mfa_layers (int): Number of last hidden states to use.
                -1 means use all.
            adapter_dim (int): Dim of adapter transform.
            dropout (float): Dropout probability.
            num_frontend_hidden_layers (int): Number of transformer
                layers in frontend.
        """
        super().__init__()

        actual_feat_dim = feat_dim
        print(
            "Adapter MFA initialized with input feature dimension: "
            f"{actual_feat_dim}"
        )

        # Frontend returns N + 1 states (input embed + N layers)
        num_available_states = num_frontend_hidden_layers + 1

        if n_mfa_layers == -1:
            self.n_mfa_layers = num_available_states
        else:
            self.n_mfa_layers = n_mfa_layers

        assert 1 <= self.n_mfa_layers <= num_available_states, (
            "Invalid n_mfa_layers "
            f"({self.n_mfa_layers}). Available: {num_available_states}"
        )

        print(
            f"Using {self.n_mfa_layers} last hidden states from frontend."
        )

        # Adapter layers for each selected hidden state
        self.adapter_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(actual_feat_dim, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.ReLU(True),
                    nn.Linear(adapter_dim, adapter_dim),
                )
                for _ in range(self.n_mfa_layers)
            ]
        )

        pooling_input_dim = adapter_dim * self.n_mfa_layers
        self.pooling = getattr(pooling_layers, pooling_func)(
            input_dim=pooling_input_dim,
            hidden_dim=adapter_dim,
        )

        pool_out_dim = self.pooling.out_dim
        self.bottleneck = nn.Linear(pool_out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, all_hidden_states: tuple):
        # Select last N states
        hidden_states_to_use = all_hidden_states[-self.n_mfa_layers :]

        # Adapter transform
        adapter_outputs = []
        for i in range(self.n_mfa_layers):
            x = self.adapter_layers[i](hidden_states_to_use[i])
            adapter_outputs.append(x)

        # Concatenate outputs
        combined_features = torch.cat(adapter_outputs, dim=-1)
        combined_features = combined_features.transpose(1, 2)

        # Pooling
        pooled_embedding = self.pooling(combined_features)

        # Dropout + Projection
        pooled_embedding = self.drop(pooled_embedding)
        embedding = self.bottleneck(pooled_embedding)

        # Dummy loss for DDP compatibility during training
        if self.training and hasattr(self, "projection"):
            dummy_loss = 0.0 * self.projection.weight.sum()
            embedding = embedding + dummy_loss

        # Return dummy tensor + embedding
        return torch.tensor(0.0, device=embedding.device), embedding
