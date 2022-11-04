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

import h5py
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
from wespeaker.utils.plda.plda_utils import norm_embeddings
from wespeaker.utils.plda.plda_utils import get_data_for_plda
from wespeaker.utils.plda.plda_utils import read_vec_scp_file


class TwoCovPLDA:

    def __init__(self, scp_file=None, utt2spk_file=None, embed_dim=256):
        if scp_file is not None:
            self.embeddings, self.embeddings_dict = \
                get_data_for_plda(scp_file, utt2spk_file)

            self.embeddings = norm_embeddings(self.embeddings)
            self.N = len(self.embeddings)
            self.mu = self.embeddings.mean(0)
            self.S = np.zeros([embed_dim, embed_dim])
            self.B = np.zeros([embed_dim, embed_dim])
            self.W = np.zeros([embed_dim, embed_dim])

            for key, mat in self.embeddings_dict.items():
                # iterate over the i-th speaker
                mat = np.vstack(mat)
                mat = norm_embeddings(mat)
                self.embeddings_dict[key] = mat
                mui = mat.mean(0)
                Si = mat.T.dot(mat)
                self.S += Si
                self.W += (mat - mui).T.dot(mat - mui)
                self.B += np.outer(mui, mui)

            self.W /= self.N
            self.B /= self.N
            # self.embeddings = self.embeddings - self.mu

            self.embed_dim = embed_dim

    def train(self, num_iters):
        """
        Implementation following paper
        Unifying Probabilistic Linear Discriminant Analysis
        Variants in Biometric Authentication
        """
        embed_dim = self.embed_dim

        T = np.zeros([embed_dim, embed_dim])
        R = np.zeros([embed_dim, embed_dim])
        Y = np.zeros(embed_dim)

        for iteration in range(1, num_iters + 1):
            print("iteration: ", iteration)
            for key, mat in self.embeddings_dict.items():
                embeddings = mat.T
                # E-step
                ni = len(mat)
                Li = inv(self.B + ni * self.W)
                Fi = embeddings.sum(1)
                gamma = self.B.dot(self.mu) + self.W.dot(Fi)
                Ey = Li.dot(gamma)
                Eyy = np.outer(Ey, Ey) + Li

                # M-step
                T = T + np.outer(Ey, Fi)
                R = R + ni * Eyy
                Y = Y + ni * Ey

            # Update the parameters
            self.mu = Y / self.N
            self.B = (R - (np.outer(self.mu, Y) + np.outer(Y, self.mu))) / \
                self.N + np.outer(self.mu, self.mu)
            self.B = inv(self.B)
            self.W = inv((self.S - (T + T.T) + R) / self.N)

    def eval_sv(self, enroll_scp, enroll_utt2spk, test_scp, trials,
                score_file):
        """
        Implementations follows
        Analysis of I-vector Length Normalization in Speaker Recognition Systems
        This function is designed for SV task
        """
        _, enroll_embeddings_dict = get_data_for_plda(enroll_scp,
                                                      enroll_utt2spk)
        test_embeddings_dict = read_vec_scp_file(test_scp)

        Stot = inv(self.W) + inv(self.B)
        Sac = inv(self.B)
        invStot = inv(Stot)
        tmp = inv(Stot - Sac.dot(invStot.dot(Sac)))
        Q = invStot - tmp  # 256 * 256
        P = invStot.dot(Sac).dot(tmp)  # 256 * 256

        enrollspks = {}
        testspks = {}
        for key, value in enroll_embeddings_dict.items():
            tmp = norm_embeddings(np.mean(value, 0))
            tmp = tmp - self.mu
            tmp.reshape(len(tmp), 1)  # 256 * 1
            enrollspks[key] = tmp.T.dot(Q).dot(tmp)
            enroll_embeddings_dict[key] = P.dot(tmp)

        for key, value in test_embeddings_dict.items():
            tmp = norm_embeddings(value) - self.mu
            tmp.reshape(len(tmp), 1)  # 256 * 1
            testspks[key] = tmp.T.dot(Q).dot(tmp)
            test_embeddings_dict[key] = tmp

        with open(score_file, 'w') as write_score:
            with open(trials, 'r') as read_trials:
                for line in tqdm(read_trials):
                    tokens = line.strip().split()
                    score = testspks[tokens[1]] + enrollspks[tokens[0]] + \
                        2.0 * test_embeddings_dict[tokens[1]].T.dot(
                        enroll_embeddings_dict[tokens[0]])
                    segs = line.strip().split()
                    output_line = ('{} {} {:.5f} {}\n'.format(
                        segs[0], segs[1], score, segs[2]))
                    write_score.write(output_line)

    def save_model(self, output_file_name):
        # assert self.validate(), "Error: wrong PLDA model format"
        print("saving the trained plda to {}".format(output_file_name))
        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("mu",
                             data=self.mu,
                             maxshape=(None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("B",
                             data=self.B,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("W",
                             data=self.W,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)

    @staticmethod
    def load_model(model_name):
        with h5py.File(model_name, "r") as f:
            plda = TwoCovPLDA()
            plda.mu = f.get("mu")[()]
            plda.B = f.get("B")[()]
            plda.W = f.get("W")[()]
            return plda
