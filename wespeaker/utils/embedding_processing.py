# Copyright (c) 2024 Johan Rohdin (rohdin@fit.vutbr.cz)
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

import re
import kaldiio
import pickle
import scipy.linalg as spl
import numpy as np
from wespeaker.utils.plda.plda_utils import get_data_for_plda


def chain_string_to_dict(chain_string=None):
    # This function converts an input string into a list and dictionary
    # structure suitable for use by the embedding processing classes below.
    # For example,
    #     "mean-subtract --scp mean1_xvector.scp | length-norm " |
    #     "| lda  --scp lda_xvector.scp --utt2spk utt2spk --dim $lda_dim "
    #     "| length-norm"
    # (The above three lines is supposed to be one long string but style
    # rules prevents it from be written that way here.)
    # becomes
    # [
    #    ['mean-subtract', {'scp': 'mean1_xvector.scp'}],
    #    ['length-norm', {}],
    #    ['lda', {'scp': 'lda_xvector.scp',
    #             'utt2spk': 'utt2spk',
    #             'dim': '100'}],
    #    ['length-norm', {}]
    # ]

    if chain_string is not None:
        links = chain_string.split('|')
    else:
        links = []

    a = []
    for l in links:

        x = l.split('--')
        method = x.pop(0)
        method = method.lstrip(' ')
        method = method.rstrip(' ')

        args_and_values = {}
        for xx in x:
            xx = re.sub("=", " ", xx)
            xx = re.sub(" +", " ", xx)
            xx = xx.lstrip(' ')
            xx = xx.rstrip(' ')
            xx = xx.split(' ')
            assert len(xx) == 2
            args_and_values[xx[0]] = xx[1]

        a.append([method, args_and_values])

    return a


class Lda:

    def compute_mean_and_lda_scatter_matrices(self,
                                              scp_file,
                                              utt2spk_file,
                                              equal_speaker_weight=False,
                                              current_chain=None):
        # equal_speaker_weight: If True, each speaker is considered equally
        # important in the calculation of the mean and scatter matrices. If
        # False, speakers are weighted by their number of utterances.
        if current_chain is None:
            current_chain = []
        _, embeddings_dict = get_data_for_plda(scp_file, utt2spk_file)
        speakers = embeddings_dict.keys()
        speaker_counts = []
        speaker_means = []
        speaker_covs = []
        n_used = 0
        n_skipped = 0
        for s in speakers:
            embd_s = current_chain(np.vstack(embeddings_dict[s]))
            count_s = embd_s.shape[0]
            # With bias=False we need at least 2 speakers, with bias=True we
            # need at least 1. But this would result in covariance matrix = 0
            # for all its elements. (This is not necessarily wrong).
            if count_s > 1:
                mean_s = np.mean(embd_s, axis=0)
                cov_s = np.cov(embd_s, rowvar=False, bias=True)
                n_used += 1
                speaker_counts.append(count_s)
                speaker_means.append(mean_s)
                speaker_covs.append(cov_s)

            else:
                n_skipped += 1

        speaker_counts = np.array(speaker_counts)
        speaker_means = np.vstack(speaker_means)
        speaker_covs = np.array(speaker_covs)
        print(
            "  #speakers: {}, #used {}, #skipped {} (only having one utterances)"
            .format(len(speakers), n_used, n_skipped))

        if equal_speaker_weight:
            mean = np.mean(speaker_means, axis=0)
            between_class_covariance = np.cov(speaker_means,
                                              rowvar=False,
                                              bias=True)
            within_class_covariance = np.sum(speaker_covs,
                                             axis=0) / len(speakers)
        else:
            mean = np.sum(speaker_counts[:, np.newaxis] * speaker_means,
                          axis=0) / np.sum(speaker_counts)
            between_class_covariance = np.cov(speaker_means,
                                              rowvar=False,
                                              bias=True,
                                              fweights=speaker_counts)
            within_class_covariance = np.sum(
                speaker_counts[:, np.newaxis, np.newaxis] * speaker_covs,
                axis=0) / np.sum(speaker_counts)

        return mean, between_class_covariance, within_class_covariance

    def __init__(self, args, current_chain=None):
        if current_chain is None:
            current_chain = []

        print(" LDA")
        scp_file = args['scp']
        utt2spk_file = args['utt2spk']
        dim = int(args['dim'])
        eps = float(args['eps']) if 'eps' in args else 1e-6

        self.m, BC, WC = self.compute_mean_and_lda_scatter_matrices(
            scp_file, utt2spk_file, current_chain=current_chain)

        E, M = spl.eigh(WC)
        # Floor the within-class covariance eigenvalues. We noticed that this
        # was done in Kaldi.
        E_floor = np.max(E) * eps
        E[E < E_floor] = E_floor
        """
        # The new within-class covariance.
        WC       = M.dot(np.diag(E).dot(M.T))
        D, lda   = spl.eigh( BC, WC )         # The output of eigh is sorted in
        self.lda = lda[:,-dim:]               # ascending order so we so we kee
        self.T1  = np.eye(self.m.shape[0])    # the "dim" last eigenvectors.
        """
        # Since we have already found the eigen decomposition of WC, we could
        # whiten it by T1 = 1 / sqrt(E), I = T1 WC T1'. So instead of solving
        # spl.eigh( BC, WC ) we can apply T1 on BC and solve
        # spl.eigh( T1 BC T1', T1 WC T1' )
        #  = spl.eigh( T1 BC T1', I )
        #  = spl.eigh( T1 BC T1')
        # as follows. However, T1 then needs to be inlcluded when transforming
        # the data. In either case, the result is that after LDA transform, the
        # data will have white WC and diagonal BC
        T1 = np.dot(np.diag(1 / np.sqrt(E)), M.T)
        BC = np.dot(np.dot(T1, BC), T1.T)
        D, lda = spl.eigh(BC)
        self.lda = np.dot(T1.T, lda[:, -dim:])

        print("  Input dimension: {}, output dimension: {},"
              " sum of all eigenvalues {:.2f}, sum of kept eigenvalues {:.2f}".
              format(len(D), dim, np.sum(D), np.sum(D[-dim:])))
        print("  All eigenvalues: {}".format(D))

    def __call__(self, embd):
        return (embd - self.m).dot(self.lda)


class Length_norm:

    def __init__(self, args=None, current_chain=None):
        pass

    def __call__(self, embd):
        embd_proc = embd.copy()
        embd_proc /= np.sqrt((embd_proc**2).sum(
            axis=1)[:, np.newaxis])  # This would make the lengths equal to one
        """
        Todo: For Kaldi compatibility we may want to add this as option as
        well as Kaldi style normalization.
        embd_proc   *= np.sqrt(embd_normed.shape[1])
        """
        return (embd_proc)


class Whitening:

    def __init__(self, args, current_chain):
        pass


class MeanSubtraction():

    def __init__(self, args, current_chain=None):
        if current_chain is None:
            current_chain = []

        e = []
        for key, vec in kaldiio.load_scp_sequential(args['scp']):
            e.append(vec)
        self.mean = np.mean(current_chain(np.vstack(e)), axis=0)

    def __call__(self, embd):
        return embd - self.mean


class EmbeddingProcessingChain:

    # This is used to map the processing steps, coming from the input
    # argument as strings, into the corresponding clases.
    string2class = {
        'lda': Lda,
        'length-norm': Length_norm,
        'whitening': Whitening,
        'mean-subtract': MeanSubtraction
    }

    def __init__(self, chain=None):
        c = chain_string_to_dict(chain)
        self.chain_of_classes = []  # This is not a great name...
        for m, a in c:
            print("Method: {}".format(m))
            print("Argument: {}".format(a))
            self.chain_of_classes.append(self.string2class[m](a, self))

    def __call__(self, embd):
        for c in self.chain_of_classes:
            embd = c(embd)
        return embd

    def save(self, path, data_format='pickle'):
        print("Saving embedding processing chain to {}".format(path))
        with open(path, 'wb') as f:
            pickle.dump(self.chain_of_classes, f)

    def load(self, path, data_format='pickle'):
        print("Loading embedding processing chain from {}".format(path))
        with open(path, 'rb') as f:
            self.chain_of_classes = pickle.load(f)

    def update_link(self, link_no_to_replace, new_link):
        nl = chain_string_to_dict(new_link)

        # For now, it is only supported to update one link. This
        # should be generalized in the future.
        assert len(nl) == 1, "Length of new chain must be one."

        m, a = nl[0]
        old_chain_of_classes = self.chain_of_classes
        self.chain_of_classes = []

        for i, ol in enumerate(old_chain_of_classes):
            if (i != link_no_to_replace):
                self.chain_of_classes.append(ol)
            else:
                print("Replacing link number {} ({}) with".format(i, ol))
                print("Method: {}".format(m))
                print("Argument: {}".format(a))
                self.chain_of_classes.append(self.string2class[m](a, self))
