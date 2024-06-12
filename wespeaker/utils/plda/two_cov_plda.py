# Copyright (c) 2022 Shuai Wang (wsstriving@gmail.com)
#               2023 Shuai Wang, Houjun Huang
#               2024 Johan Rohdin (rohdin@fit.vutbr.cz)
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

import collections
import math
import h5py
import numpy as np
import scipy.linalg as spl
from numpy.linalg import inv
from tqdm import tqdm
from wespeaker.utils.plda.kaldi_utils import read_plda

from wespeaker.utils.plda.plda_utils import compute_normalizing_transform
from wespeaker.utils.plda.plda_utils import get_data_for_plda
from wespeaker.utils.plda.plda_utils import norm_embeddings
from wespeaker.utils.plda.plda_utils import read_vec_scp_file
from wespeaker.utils.plda.plda_utils import sort_svd

M_LOG_2PI = 1.8378770664093454835606594728112

ClassInfo = collections.namedtuple('ClassInfo',
                                   ['weight', 'num_example', 'mu'])


class PldaStats(object):

    def __init__(self, dim):
        self.dim = dim
        self.num_example, self.num_classes = 0, 0
        self.class_weight, self.example_weight = 0, 0,
        self.sum_, self.offset_scatter = np.zeros(dim), np.zeros((dim, dim))
        self.classinfo = []

    def add_samples(self, weight, spk_embeddings):
        """
        Add samples of a certain speaker to the PLDA stats
        :param weight: class_weight, default set to 1.
        :param spk_embeddings: All embedding samples from a certain speaker
        :return:
        """
        n = spk_embeddings.shape[0]
        mean = np.mean(spk_embeddings, axis=0)
        tmp = spk_embeddings - mean
        self.offset_scatter += weight * np.matmul(tmp.T, tmp)
        self.classinfo.append(ClassInfo(weight, n, mean))
        self.num_example += n
        self.num_classes += 1
        self.class_weight += weight
        self.example_weight += weight * n
        self.sum_ += weight * mean


class TwoCovPLDA:

    def __init__(self,
                 scp_file=None,
                 utt2spk_file=None,
                 embed_dim=256,
                 subtract_train_set_mean=False,
                 normalize_length=False):
        self.subtract_train_set_mean = subtract_train_set_mean
        self.normalize_length = normalize_length
        self.dim = embed_dim
        self.mu = np.zeros(self.dim)
        # The transform which whitens the within- and
        # diagonalizes the across-class covariance matrix
        self.transform = np.zeros((self.dim, self.dim))
        # The diagonal of the across-class covariance in the transformed space
        self.psi = np.zeros(self.dim)
        self.offset = np.zeros(self.dim)
        self.stats = PldaStats(self.dim)
        self.B = np.eye(self.dim)
        self.B_stats = np.zeros((self.dim, self.dim))
        self.B_count = 0
        self.W = np.eye(self.dim)
        self.W_stats = np.zeros((self.dim, self.dim))
        self.W_count = 0
        if scp_file is not None:
            samples, self.embeddings_dict = get_data_for_plda(
                scp_file, utt2spk_file)
            if subtract_train_set_mean:
                train_mean_vec = samples.mean(0)
            else:
                train_mean_vec = np.zeros(embed_dim)
            for key, mat in self.embeddings_dict.items():
                mat = np.vstack(mat)
                mat = mat - train_mean_vec
                if self.normalize_length:
                    mat = norm_embeddings(mat)
                self.stats.add_samples(1.0, mat)
            self.mu = self.stats.sum_ / self.stats.class_weight

    def train(self, num_em_iters):
        for i in range(num_em_iters):
            print("Plda estimation %d of %d" % (i, num_em_iters))
            self.em_one_iter()
        self.get_output()

    def em_one_iter(self):
        self.B_stats, self.B_count = np.zeros(
            (self.stats.dim, self.stats.dim)), 0
        self.W_stats, self.W_count = np.zeros(
            (self.stats.dim, self.stats.dim)), 0
        self.W_stats += self.stats.offset_scatter
        self.W_count += self.stats.example_weight - self.stats.class_weight
        B_inv = inv(self.B)
        W_inv = inv(self.W)
        for i in range(self.stats.num_classes):
            info = self.stats.classinfo[i]
            m = info.mu - self.stats.sum_ / self.stats.class_weight
            weight = info.weight
            n = info.num_example
            mix_var = inv(B_inv + n * W_inv)
            w = np.matmul(mix_var, n * np.matmul(W_inv, m))
            m_w = m - w
            self.B_stats += weight * (mix_var + np.outer(w, w))
            self.B_count += weight
            self.W_stats += weight * n * (mix_var + np.outer(m_w, m_w))
            self.W_count += weight

        self.W = self.W_stats / self.W_count
        self.B = self.B_stats / self.B_count
        self.W = 0.5 * (self.W + self.W.T)
        self.B = 0.5 * (self.B + self.B.T)

        print("W_count:", self.W_count, "Trace of W:", np.trace(self.W))
        print("B_count:", self.B_count, "Trace of B:", np.trace(self.B))

    def get_output(self):
        self.mu = self.stats.sum_ / self.stats.class_weight
        transform1 = compute_normalizing_transform(self.W)
        B_proj = np.matmul(transform1, self.B)
        B_proj = np.matmul(B_proj, transform1.T)
        s, U = np.linalg.eigh(B_proj)
        s = np.where(s > 0.0, s, 0.0)
        s, U = sort_svd(s, U)

        self.transform = np.matmul(U.T, transform1)
        self.psi = s
        self.offset = np.zeros(self.dim)
        self.offset = -1.0 * np.matmul(self.transform, self.mu)

    def transform_embedding(self, embedding):
        transformed_embedding = np.matmul(self.transform, embedding)
        transformed_embedding += self.offset
        normalization_factor = math.sqrt(
            self.dim) / np.linalg.norm(transformed_embedding)
        if self.normalize_length:
            transformed_embedding = normalization_factor * transformed_embedding
        return transformed_embedding

    def log_likelihood_ratio(self, transformed_train_embedding,
                             transformed_test_embedding, n):
        mean = n * self.psi / (n * self.psi +
                               1.0) * transformed_train_embedding
        variance = 1.0 + self.psi / (n * self.psi + 1.0)
        logdet = np.sum(np.log(variance))
        sqdiff = transformed_test_embedding - mean
        sqdiff = np.power(sqdiff, 2.0)
        variance = 1.0 / variance
        loglike_given_class = -0.5 * (logdet + M_LOG_2PI * self.dim +
                                      np.dot(sqdiff, variance))
        sqdiff = transformed_test_embedding
        sqdiff = np.power(sqdiff, 2.0)
        variance = self.psi + 1.0
        logdet = np.sum(np.log(variance))
        variance = 1.0 / variance
        loglike_without_class = -0.5 * (logdet + M_LOG_2PI * self.dim +
                                        np.dot(sqdiff, variance))
        loglike_ratio = loglike_given_class - loglike_without_class
        return loglike_ratio

    def eval_sv(self,
                enroll_scp,
                enroll_utt2spk,
                test_scp,
                trials,
                score_file,
                multisession_avg=True,
                indomain_scp=None):
        """
        Caculate the plda score
        :param enroll_scp:
        :param enroll_utt2spk:
        :param test_scp:
        :param trials:
        :param score_file:
        :param indomain_scp:
        :return:
        """
        _, enroll_embeddings_dict = get_data_for_plda(enroll_scp,
                                                      enroll_utt2spk)
        test_embeddings_dict = read_vec_scp_file(test_scp)

        if indomain_scp is not None:
            indomain_embeddings_dict = read_vec_scp_file(indomain_scp)
            mean_vec = np.vstack(list(
                indomain_embeddings_dict.values())).mean(0)
        else:
            mean_vec = np.zeros(self.dim)

        enrollspks = {}
        testspks = {}
        enrollcounts = {}
        for key, value in enroll_embeddings_dict.items():
            if multisession_avg:
                enrollcounts[key] = 1
            else:
                enrollcounts[key] = len(value)
            value = np.vstack(value)
            value = value - mean_vec  # Shuai

            # Normalize length
            # It is questionable whether this should be applied
            # after speaker mean in case of multisession scoring.
            if self.normalize_length:
                tmp = norm_embeddings(np.mean(value, 0))

            else:
                tmp = np.mean(value, 0)
            tmp = self.transform_embedding(tmp)
            enrollspks[key] = tmp

        for key, value in test_embeddings_dict.items():
            value = value - mean_vec  # Shuai
            if self.normalize_length:
                tmp = norm_embeddings(value)
            else:
                tmp = value
            tmp = self.transform_embedding(tmp)
            testspks[key] = tmp

        with open(score_file, 'w') as write_score:
            with open(trials, 'r') as read_trials:
                for line in tqdm(read_trials):
                    tokens = line.strip().split()
                    score = self.log_likelihood_ratio(enrollspks[tokens[0]],
                                                      testspks[tokens[1]],
                                                      enrollcounts[tokens[0]])
                    segs = line.strip().split()
                    output_line = ('{} {} {:.5f} {}\n'.format(
                        segs[0], segs[1], score, segs[2]))
                    write_score.write(output_line)

    def adapt(self, adapt_scp, ac_scale=0.5, wc_scale=0.5):
        # Implemented by the BUT speech group
        # plda = load_model(model_path, from_kaldi=from_kaldi)
        adp_data = np.array(list(read_vec_scp_file(adapt_scp).values()))
        mean_vec = adp_data.mean(0)
        adp_data = adp_data - mean_vec
        if self.normalize_length:
            adp_data = norm_embeddings(adp_data)

        plda_mean, plda_trans, plda_psi = self.mu, self.transform, self.psi
        W = inv(plda_trans.T.dot(plda_trans))
        W = (W + W.T) / 2
        B = np.linalg.inv((plda_trans.T / plda_psi).dot(plda_trans))
        B = (B + B.T) / 2
        T = B + W
        # adp_data = np.vstack(self.xvect)
        # Covariance of the adaptation data.
        data_cov = np.cov(adp_data.T)
        [v, e] = spl.eigh(data_cov, (T + T.T) / 2)
        iet = np.linalg.inv(e.T)
        excess = iet[:, v > 1].dot(np.diag(np.sqrt(v[v > 1] - 1)))
        V_adp = excess * np.sqrt(ac_scale)
        B_adp = B + V_adp.dot(V_adp.T)
        U_adp = excess * np.sqrt(wc_scale)
        W_adp = W + U_adp.dot(U_adp.T)
        mu_adp = np.mean(adp_data, axis=0)
        mu, A, B = mu_adp, (B_adp + B_adp.T) / 2.0, (W_adp + W_adp.T) / 2.0
        eps = 1e-9
        [D, V] = np.linalg.eigh(B)
        D = np.diag(1.0 / np.sqrt(D + eps))
        # First transform
        T1 = np.dot(D, V.T)
        # This should equal the identity matrix
        B1 = np.dot(np.dot(T1, B), T1.T)
        A1 = np.dot(np.dot(T1, A), T1.T)
        # Second transform is given by T2.T * (.) * T2
        [D, T2] = np.linalg.eigh(A1)
        # Joint transform
        T = np.dot(T2.T, T1)
        # Transform the matrices
        A2 = np.dot(np.dot(T, A), T.T)
        B2 = np.dot(np.dot(T, B), T.T)
        plda_trans, plda_psi, X = T, np.diag(A2), B2

        adapt_plda = TwoCovPLDA()
        adapt_plda.mu = mu
        adapt_plda.transform = plda_trans
        adapt_plda.psi = plda_psi
        adapt_plda.offset = -1.0 * np.matmul(adapt_plda.transform,
                                             adapt_plda.mu)

        return adapt_plda

    def save_model(self, output_file_name):
        print("saving the trained plda to {}".format(output_file_name))
        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("mu",
                             data=self.mu,
                             maxshape=(None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("transform",
                             data=self.transform,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("psi",
                             data=self.psi,
                             maxshape=(None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("offset",
                             data=self.offset,
                             maxshape=(None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("normalize_length",
                             data=int(self.normalize_length),
                             maxshape=(None))
            f.create_dataset("subtract_train_set_mean",
                             data=int(self.subtract_train_set_mean),
                             maxshape=(None))

    @staticmethod
    def load_model(model_name, from_kaldi=False):
        plda = TwoCovPLDA()
        if from_kaldi:
            plda.mu, plda.transform, plda.psi = read_plda(model_name)
            plda.offset = np.zeros(plda.mu.shape[0])
            plda.offset = -1.0 * np.matmul(plda.transform, plda.mu)
        else:
            with h5py.File(model_name, "r") as f:
                plda.mu = f.get("mu")[()]
                plda.transform = f.get("transform")[()]
                plda.psi = f.get("psi")[()]
                plda.offset = f.get("offset")[()]
                plda.normalize_length = bool(f.get("normalize_length")[()])
                plda.subtract_train_set_mean = bool(
                    f.get("subtract_train_set_mean")[()])
                print("PLDA normalize length is {}.".format(
                    plda.normalize_length))
                print("PLDA subtract_train_set_mean is {}.".format(
                    plda.subtract_train_set_mean))

        plda.dim = plda.mu.shape[0]
        return plda
