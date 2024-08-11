# Copyright (c) 2023 Xu Xiang
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
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import concurrent.futures
from collections import OrderedDict

import numpy as np

import kaldiio
import umap
import hdbscan
import pahc


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--scp', required=True, help='embedding scp')
    parser.add_argument('--output', required=True, help='output label file')
    parser.add_argument('--n_neighbors', required=False, default=16,
                        help="The size of the local neighborhood UMAP "
                             "will look at when attempting to learn "
                             "the manifold structure of the data. "
                             "This means that low values of n_neighbors "
                             "will force UMAP to concentrate on "
                             "very local structure (potentially to "
                             "the detriment of the big picture), "
                             "while large values will push UMAP to "
                             "look at larger neighborhoods of each point "
                             "when estimating the manifold structure of "
                             "the data, losing fine detail structure for "
                             "the sake of getting the broader of the data.")
    parser.add_argument('--min_dist', required=False, default=0.1,
                        help="The minimum distance between points in "
                             "the low dimensional representation.")
    args = parser.parse_args()
    return args


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


def cluster(embeddings):
    # Fallback
    if len(embeddings) <= 2:
        return [0] * len(embeddings)

    n_neighbors, min_dist = int(args.n_neighbors), float(args.min_dist)

    umap_embeddings = umap.UMAP(n_components=min(32, len(embeddings) - 2),
                                metric='cosine',
                                n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                random_state=2020,
                                n_jobs=1).fit_transform(np.array(embeddings))

    labels = hdbscan.HDBSCAN(core_dist_n_jobs=1,
                             allow_single_cluster=True,
                             min_cluster_size=4).fit_predict(umap_embeddings)

    labels = pahc.PAHC(merge_cutoff=0.3,
                       min_cluster_size=3,
                       absorb_cutoff=0.0).fit_predict(labels, embeddings)
    return labels


if __name__ == '__main__':
    args = get_args()

    subsegs_list, embeddings_list = read_emb(args.scp)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        with open(args.output, 'w') as fd:
            for (subsegs, labels) in zip(subsegs_list,
                                         executor.map(cluster,
                                                      embeddings_list)):
                [print(subseg,
                       label,
                       file=fd) for (subseg, label) in zip(subsegs, labels)]
