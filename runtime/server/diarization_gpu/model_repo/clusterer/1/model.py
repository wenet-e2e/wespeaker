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

import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
import json
import cupy as cp
import numpy as np
from cuml.cluster import KMeans as cuKM


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance
          *                           device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "LABELS")
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def cluster_gpu(self,
                    embeddings,
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
            kmeans_float = cuKM(n_clusters=k, n_init=10, random_state=10)
            kmeans_float.fit(cp.asarray(data))
            return kmeans_float.labels_

        # Fallback for trivial cases
        if len(embeddings) <= 2:
            return [0] * len(embeddings)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        # Prune matrix with p interval
        pruned_similarity_matrix = prune(similarity_matrix, p)
        # Compute Laplacian
        laplacian_matrix = laplacian(pruned_similarity_matrix)
        # Compute spectral embeddings
        spectral_embeddings = spectral(laplacian_matrix, num_spks,
                                       min_num_spks, max_num_spks)
        # Assign class labels
        labels = kmeans(spectral_embeddings)

        return labels

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse.
          The length of this list must be the same as `requests`
        """
        batch_count = []
        total_embd = []

        responses = []
        for request in requests:
            # the requests will all have the same shape
            # different shape request will be
            # separated by triton inference server
            input0 = pb_utils.get_input_tensor_by_name(request, "EMBEDDINGS")
            cur_b_embd = from_dlpack(input0.to_dlpack())
            cur_batch = cur_b_embd.shape[0]
            batch_count.append(cur_batch)

            for embds in cur_b_embd:
                total_embd.append(embds.to(self.device))

        labels_list = []
        for embds in total_embd:
            res = self.cluster_gpu(cp.asarray(embds))
            labels_list.append(cp.asnumpy(res))

        idx = 0
        for b in batch_count:
            batch_labels = np.array(labels_list[idx:idx + b])
            idx += b
            out0 = pb_utils.Tensor("LABELS",
                                   batch_labels.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0])
            responses.append(inference_response)
        return responses
