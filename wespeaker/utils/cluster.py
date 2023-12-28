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

import numpy as np
import scipy.linalg
from sklearn.cluster._kmeans import k_means


def cluster(embeddings, p=.01, num_spks=None, min_num_spks=1, max_num_spks=20):
    # Define utility functions
    def cosine_similarity(M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return 0.5 * (1.0 + np.dot(M, M.T))

    def prune(M, p):
        m = M.shape[0]
        if m < 1000:
            n = max(m - 10, 2)
        else:
            n = int((1.0 - p) * m)

        for i in range(m):
            indexes = np.argsort(M[i, :])
            low_indexes, high_indexes = indexes[0:n], indexes[n:m]
            M[i, low_indexes] = 0.0
            M[i, high_indexes] = 1.0
        return 0.5 * (M + M.T)

    def laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0.0
        D = np.diag(np.sum(np.abs(M), axis=1))
        return D - M

    def spectral(M, num_spks, min_num_spks, max_num_spks):
        eig_values, eig_vectors = scipy.linalg.eigh(M)
        num_spks = num_spks if num_spks is not None \
            else np.argmax(np.diff(eig_values[:max_num_spks + 1])) + 1
        num_spks = max(num_spks, min_num_spks)
        return eig_vectors[:, :num_spks]

    def kmeans(data):
        k = data.shape[1]
        # centroids, labels = scipy.cluster.vq.kmeans2(data, k, minit='++')
        _, labels, _ = k_means(data, k, random_state=None, n_init=10)
        return labels

    # Fallback for trivial cases
    if len(embeddings) <= 2:
        return [0] * len(embeddings)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(np.array(embeddings))
    # Prune matrix with p interval
    pruned_similarity_matrix = prune(similarity_matrix, p)
    # Compute Laplacian
    laplacian_matrix = laplacian(pruned_similarity_matrix)
    # Compute spectral embeddings
    spectral_embeddings = spectral(laplacian_matrix, num_spks, min_num_spks,
                                   max_num_spks)
    # Assign class labels
    labels = kmeans(spectral_embeddings)

    return labels
