# Copyright (c) 2024 Yiyang Zhao (zhaoyy22@mails.tsinghua.edu.cn)
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

import torch
from torch import nn

import wespeaker.models.pooling_layers as pooling_layers


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=True,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


class whisper_PMFA(torch.nn.Module):
    def __init__(
            self,
            output_size=1280,
            embedding_dim=192,
            pooling_func='ASTP',
            global_context_att=True):
        super(whisper_PMFA, self).__init__()
        self.pooling = getattr(pooling_layers, pooling_func)(
            in_dim=output_size, global_context_att=global_context_att)
        self.bn = BatchNorm1d(input_size=output_size * 2)
        self.fc = torch.nn.Linear(output_size * 2, embedding_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = x.unsqueeze(-1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x


def whisper_PMFA_large_v2(feat_dim, embed_dim):
    return whisper_PMFA(output_size=feat_dim,
                        embedding_dim=embed_dim
                        )
