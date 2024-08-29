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
from collections import OrderedDict, defaultdict
import functools
import heapq

import numpy as np

import kaldiio
import umap
import hdbscan


class PAHC:
    def __init__(self, merge_cutoff=0.3, min_cluster_size=3, absorb_cutoff=0.0):
        self.merge_cutoff = merge_cutoff
        self.min_cluster_size = min_cluster_size
        self.absorb_cutoff = absorb_cutoff

    def fit_predict(self, labels, embeddings):
        self.initialize(labels, embeddings)
        self.merge_cluster()
        self.absorb_cluster()
        labels = self.relabel_cluster()
        return labels

    def initialize(self, labels, embeddings):
        self.labels = labels
        self.embeddings = embeddings
        self.active_clusters = set([])
        self.label_map = {}
        self.cost_map = {}
        self.heap = []
        self.next_index = -1

        self.build_label_map()
        self.build_cost_map()

    def merge_cluster(self):
        while self.heap:
            _, (i, j) = heapq.heappop(self.heap)
            if i in self.active_clusters and j in self.active_clusters:
                self.merge(i, j)

    def absorb_cluster(self):
        minor_clusters = set()
        major_clusters = set()
        for k, indexes in self.label_map.items():
            if len(indexes) < self.min_cluster_size:
                minor_clusters.add(k)
            else:
                major_clusters.add(k)

        if len(major_clusters) > 0:
            for i in minor_clusters:
                max_cost = -np.inf
                for j in major_clusters:
                    pair = (i, j) if i < j else (j, i)
                    i_indexes, j_indexes = self.label_map[i], self.label_map[j]
                    factor = len(i_indexes) * len(j_indexes)
                    normalized_cost = self.cost_map[pair] / factor
                    if normalized_cost > max_cost:
                        max_cost = normalized_cost
                        closest_cluster = j
                if max_cost >= self.absorb_cutoff:
                    self.label_map[closest_cluster].extend(self.label_map[i])
                    self.eliminate(i)

    def relabel_cluster(self):
        labels = [-1] * len(self.labels)

        for label, indexes in self.label_map.items():
            for index in indexes:
                labels[index] = label
        i = 0
        label_to_label = {}
        for label in labels:
            if label not in label_to_label:
                label_to_label[label] = i
                i += 1
        for i in range(len(labels)):
            labels[i] = label_to_label[labels[i]]
        return labels

    def eliminate(self, i):
        del self.label_map[i]
        self.active_clusters.remove(i)

    def build_label_map(self):
        self.label_map = defaultdict(list)

        for i, label in enumerate(self.labels):
            self.label_map[label].append(i)

        self.num_labeled = len(self.label_map)

        if -1 in self.label_map:
            self.num_labeled -= 1
            for i, j in zip(range(self.num_labeled,
                                  self.num_labeled + len(self.label_map[-1])),
                            self.label_map[-1]):
                self.label_map[i].append(j)
            del self.label_map[-1]

    def build_cost_map(self):
        N = len(self.label_map)
        self.active_clusters = set(range(N))
        self.next_index = N

        for i in range(N):
            for j in range(i + 1, N):
                i_indexes, j_indexes = self.label_map[i], self.label_map[j]

                if i < self.num_labeled and j < self.num_labeled:
                    self.cost_map[(i, j)] = -np.inf
                    continue

                self.cost_map[(i, j)] = self.compute_cost(i_indexes, j_indexes)

                factor = len(i_indexes) * len(j_indexes)
                normalized_cost = self.cost_map[(i, j)] / factor
                if normalized_cost >= self.merge_cutoff:
                    heapq.heappush(self.heap, (-normalized_cost, (i, j)))

    def compute_cost(self, i_indexes, j_indexes):
        i_embedding = sum([
            self.l2norm(self.embeddings[i_index]) for i_index in i_indexes])
        j_embedding = sum([
            self.l2norm(self.embeddings[j_index]) for j_index in j_indexes])
        return np.dot(i_embedding, j_embedding)

    def merge(self, i, j):
        i_indexes, j_indexes = self.label_map[i], self.label_map[j]

        for k, _ in self.label_map.items():
            if k == i or k == j:
                continue
            pair1 = (k, i) if k < i else (i, k)
            pair2 = (k, j) if k < j else (j, k)
            cost = self.cost_map[pair1] + self.cost_map[pair2]
            self.cost_map[(k, self.next_index)] = cost

            factor = (len(i_indexes) + len(j_indexes)) * len(self.label_map[k])
            normalized_cost = cost / factor
            if normalized_cost >= self.merge_cutoff:
                heapq.heappush(self.heap, (-normalized_cost,
                                           (k, self.next_index)))

        self.label_map[self.next_index] = i_indexes + j_indexes
        self.active_clusters.add(self.next_index)
        self.eliminate(i)
        self.eliminate(j)
        self.next_index += 1

    def l2norm(self, x, axis=0, keepdims=True):
        return x / np.linalg.norm(x, axis=axis, keepdims=keepdims)


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
    parser.add_argument('--min_dist', required=False, default=0.05,
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


def cluster(embeddings, n_neighbors=16, min_dist=0.05):
    # Fallback
    if len(embeddings) <= 2:
        return [0] * len(embeddings)

    umap_embeddings = umap.UMAP(n_components=min(32, len(embeddings) - 2),
                                metric='cosine',
                                n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                random_state=2023,
                                n_jobs=1).fit_transform(np.array(embeddings))

    labels = hdbscan.HDBSCAN(allow_single_cluster=True,
                             min_cluster_size=4,
                             approx_min_span_tree=False,
                             core_dist_n_jobs=1).fit_predict(umap_embeddings)

    labels = PAHC(merge_cutoff=0.3,
                  min_cluster_size=3,
                  absorb_cutoff=0.0).fit_predict(labels, embeddings)
    return labels


if __name__ == '__main__':
    args = get_args()

    subsegs_list, embeddings_list = read_emb(args.scp)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_neighbors, min_dist = int(args.n_neighbors), float(args.min_dist)

    run_cluster = functools.partial(cluster,
                                    n_neighbors=n_neighbors,
                                    min_dist=min_dist)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        with open(args.output, 'w') as fd:
            for (subsegs, labels) in zip(subsegs_list,
                                         executor.map(run_cluster,
                                                      embeddings_list)):
                [print(subseg,
                       label,
                       file=fd) for (subseg, label) in zip(subsegs, labels)]
