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


import heapq
from collections import defaultdict

import numpy as np


def l2norm(x, axis=0, keepdims=True):
    return x / np.linalg.norm(x, axis=axis, keepdims=keepdims)


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
            l2norm(self.embeddings[i_index]) for i_index in i_indexes])
        j_embedding = sum([
            l2norm(self.embeddings[j_index]) for j_index in j_indexes])
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
