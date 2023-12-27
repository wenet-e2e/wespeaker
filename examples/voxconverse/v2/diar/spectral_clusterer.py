# Copyright (c) 2022 Xu Xiang
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

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from collections import OrderedDict
import concurrent.futures as cf
import kaldiio

import numpy as np
from wespeaker.utils.utils import validate_path
from wespeaker.utils.cluster import cluster


def read_emb(scp):

    emb_dict = OrderedDict()
    for sub_seg_id, emb in kaldiio.load_scp_sequential(scp):
        utt = sub_seg_id.split('-')[0]
        if utt not in emb_dict:
            emb_dict[utt] = {}
            emb_dict[utt]['sub_seg'] = []
            emb_dict[utt]['embs'] = []
        emb_dict[utt]['sub_seg'].append(sub_seg_id)
        emb_dict[utt]['embs'].append(emb)

    subsegs_list = []
    embeddings_list = []

    for utt, utt_emb_dict in emb_dict.items():
        subsegs_list.append(utt_emb_dict['sub_seg'])
        embeddings_list.append(np.stack(utt_emb_dict['embs']))

    return subsegs_list, embeddings_list


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--scp', required=True, help='embedding scp')
    parser.add_argument('--output', required=True, help='output label file')
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    subsegs_list, embeddings_list = read_emb(args.scp)
    validate_path(args.output)

    with cf.ProcessPoolExecutor() as executor, open(args.output, 'w') as f:
        for (subsegs, labels) in zip(subsegs_list,
                                     executor.map(cluster, embeddings_list)):
            [
                print(subseg, label, file=f)
                for (subseg, label) in zip(subsegs, labels)
            ]


if __name__ == '__main__':
    main()
