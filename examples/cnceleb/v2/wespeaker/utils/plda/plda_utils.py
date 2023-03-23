# Copyright (c) 2022 Shuai Wang (wsstriving@gmail.com)
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
import kaldiio


def read_vec_scp_file(scp_file):
    """
    Read the pre-extracted kaldi-format speaker embeddings.
    :param scp_file: path to xvector.scp
    :return: dict {wav_name: embedding}
    """
    samples_dict = {}
    for key, vec in kaldiio.load_scp_sequential(scp_file):
        samples_dict[key] = vec
    return samples_dict


def read_label_file(label_file):
    """
    Read the utt2spk file
    :param label_file: the path to utt2spk
    :return: dict {wav_name: spk_id}
    """
    labels_dict = {}
    with open(label_file, 'r') as fin:
        for line in fin:
            tokens = line.strip().split()
            labels_dict[tokens[0]] = tokens[1]
    return labels_dict


def norm_embeddings(embeddings):
    if len(embeddings.shape) == 2:
        return (embeddings.transpose() /
                np.linalg.norm(embeddings, axis=1)).transpose()
    elif len(embeddings.shape) == 1:
        return embeddings / np.linalg.norm(embeddings)


def get_data_for_plda(scp_file, utt2spk_file):
    samples_dict = read_vec_scp_file(scp_file)
    labels_dict = read_label_file(utt2spk_file)
    samples = []
    model_dict = {}
    for key, vec in samples_dict.items():
        samples.append(vec)
        label = labels_dict[key]
        if label in model_dict.keys():
            model_dict[label].append(vec)
        else:
            model_dict[label] = [vec]
    return np.vstack(samples), model_dict
