# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
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

import numpy as np
import random
from scipy import signal


def get_random_chunk(data, chunk_len):
    # chunking: randomly select a range of size min(chunk_len, len).
    data_len = len(data)
    data_shape = data.shape
    adjust_chunk_len = min(data_len, chunk_len)
    chunk_start = random.randint(0, data_len - adjust_chunk_len)

    data = data[chunk_start:chunk_start + adjust_chunk_len]
    # padding if needed
    if adjust_chunk_len < chunk_len:
        chunk_shape = chunk_len if len(data_shape) == 1 else (chunk_len,
                                                              data.shape[1])
        data = np.resize(data, chunk_shape)  # repeating

    return data


def spec_augmentation(x,
                      num_t_mask=1,
                      num_f_mask=1,
                      max_t=10,
                      max_f=8,
                      prob=0.5):
    """ do spec augmentation on x

    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask

    Returns:
        augmented feature (x)
    """
    if random.random() > prob:
        return x

    y = x  # np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]

    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y


def speed_perturb(x, speed_perturb_idx=0):
    speed_list = [1.0, 0.9, 1.1]
    speed = speed_list[speed_perturb_idx]

    x = x.astype(np.float32)
    y = signal.resample(x, int(len(x) / speed))

    return y
