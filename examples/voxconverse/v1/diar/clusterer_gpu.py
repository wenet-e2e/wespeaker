# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cupy as cp
from cuml.cluster import KMeans as cuKM
import numpy as np
from timeit import default_timer as timer
from clusterer import get_args, compute_embeddings
import torch


def cluster_gpu(embeddings,
                p=.01,
                num_spks=None,
                min_num_spks=1,
                max_num_spks=20):
    # Define utility functions
    def cosine_similarity(M):
        M = M / cp.linalg.norm(M, axis=1, keepdims=True)
        return 0.5 * (1.0 + cp.dot(M, M.T))

    def prune(M, p):
        m = M.shape[0]
        if m < 1000:
            n = max(m - 10, 2)
        else:
            n = int((1.0 - p) * m)
        for i in range(m):
            indexes = cp.argsort(M[i, :])
            low_indexes, high_indexes = indexes[0:n], indexes[n:m]
            M[i, low_indexes] = 0.0
            M[i, high_indexes] = 1.0
        return 0.5 * (M + M.T)

    def laplacian(M):
        M[cp.diag_indices(M.shape[0])] = 0.0
        D = cp.diag(cp.sum(cp.abs(M), axis=1))
        return D - M

    def spectral(M, num_spks, min_num_spks, max_num_spks):
        eig_values, eig_vectors = cp.linalg.eigh(M)
        num_spks = num_spks if num_spks is not None \
            else cp.argmax(cp.diff(eig_values[:max_num_spks + 1])) + 1
        num_spks = max(num_spks, min_num_spks)
        return eig_vectors[:, :num_spks]

    def kmeans(data):
        k = data.shape[1]
        kmeans_float = cuKM(n_clusters=k, n_init=10)
        kmeans_float.fit(cp.asarray(data))
        return kmeans_float.labels_

    # Fallback for trivial cases
    if len(embeddings) <= 2:
        return [0] * len(embeddings)

    # How to specify the cuda device?
    # with cp.cuda.Device(1):
    #     embeddings = cp.array(embeddings)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
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


def test_time():
    a = np.random.rand(1000, 256)

    def with_cuda(x, count):
        for _ in range(count):
            l = cluster_gpu(x)
        return l

    for c in [1, 10, 100, 1000, 10000]:
        print(c)
        data = cp.asarray(a)
        start = timer()
        r = with_cuda(data, c)
        cp.cuda.Device().synchronize()
        elapsed_time = timer() - start
        print("GPU Time: {}".format(elapsed_time))
        start = timer()
        r = with_cpu(a, c)
        elapsed_time = timer() - start
        print("CPU Time: {}".format(elapsed_time))


def main():
    args = get_args()
    print('Segmenting and extracting speaker embeddings')
    subsegs_list, embeddings_list = compute_embeddings(args.scp, args.segments,
                                                       args.source,
                                                       args.device,
                                                       args.batch_size)
    print('Embedding extraction finished')
    print('Start GPU Clustering')

    # Use the following part to do GPU Clustering
    labels_list = []
    with open(args.output, 'w') as f:
        for i in embeddings_list:
            labels_list.append(cluster_gpu(cp.asarray(i)))
        for (subsegs, labels) in zip(subsegs_list, labels_list):
            [
                print(subseg, label, file=f)
                for (subseg, label) in zip(subsegs, labels)
            ]


if __name__ == '__main__':
    # You can use the test_time() function
    # to calculate the GPU vs CPU clustering speed
    torch.set_num_threads(1)

    main()
