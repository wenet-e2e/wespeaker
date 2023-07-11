# Copyright (c) 2023, Zhengyang Chen (chenzhengyang117@gmail.com)
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, encoder, embed_dim=256, T=0.07, mlp=False, n_views=2):
        """
        T: softmax temperature (default: 0.07)
        n_views: number of views for each sample
        """
        super(SimCLR, self).__init__()

        self.T = T
        self.n_views = n_views
        self.encoder = encoder

        if mlp:
            self.encoder.add_module(
                "mlp",
                nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                              nn.Linear(embed_dim, embed_dim)))
        else:
            self.encoder.add_module("mlp", nn.Sequential())

    def prepare_for_info_nce_loss(self, features):
        '''
        Input:
            features: (self.n_views * bs, embed_dim)
        Return:
            logits: (self.n_views * bs, self.n_views * bs - 1)
            labels (torch.long): (self.n_views * bs)

        '''

        bs = features.shape[0] // self.n_views
        # labels: (self.n_views * bs)
        labels = torch.cat([torch.arange(bs) for _ in range(self.n_views)],
                           dim=0)
        # labels: (self.n_views * bs, self.n_views * bs)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        # similarity_matrix: (self.n_views * bs, self.n_views * bs)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0],
                             dtype=torch.long).to(features.device)

        logits = logits / self.T
        return logits, labels

    def forward(self, input_q, input_k):
        """
        Input:
            input_q: a batch of query inputs
            input_k: a batch of key inputs
        Output:
            logits, targets
        """

        combine_input = torch.cat((input_q, input_k), dim=0)
        features = self.encoder(combine_input)
        features = self.encoder.mlp(features)

        logits, labels = self.prepare_for_info_nce_loss(features)

        return logits, labels
