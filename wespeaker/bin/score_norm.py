# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

import logging
import os

import fire
import kaldiio
import numpy as np
from tqdm import tqdm

from wespeaker.utils.file_utils import read_table


def get_mean_std(emb, cohort, top_n):
    emb = emb / np.sqrt(np.sum(emb**2, axis=1, keepdims=True))
    cohort = cohort / np.sqrt(np.sum(cohort**2, axis=1, keepdims=True))
    emb_cohort_score = np.matmul(emb, cohort.T)
    emb_cohort_score = np.sort(emb_cohort_score, axis=1)[:, ::-1]
    emb_cohort_score_topn = emb_cohort_score[:, :top_n]

    emb_mean = np.mean(emb_cohort_score_topn, axis=1)
    emb_std = np.std(emb_cohort_score_topn, axis=1)

    return emb_mean, emb_std


def split_embedding(utt_list, emb_scp, mean_vec):
    embs = []
    utt2idx = {}
    utt2emb = {}
    for utt, emb in kaldiio.load_scp_sequential(emb_scp):
        emb = emb - mean_vec
        utt2emb[utt] = emb

    for utt in utt_list:
        embs.append(utt2emb[utt])
        utt2idx[utt] = len(embs) - 1

    return np.array(embs), utt2idx


def main(score_norm_method,
         top_n,
         trial_score_file,
         score_norm_file,
         cohort_emb_scp,
         eval_emb_scp,
         mean_vec_path=None):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # get embedding
    if not mean_vec_path:
        print("Do not do mean normalization for evaluation embeddings.")
        mean_vec = 0.0
    else:
        assert os.path.exists(
            mean_vec_path), "mean_vec file ({}) does not exist !!!".format(
                mean_vec_path)
        mean_vec = np.load(mean_vec_path)

    # get embedding
    logging.info('get embedding ...')

    enroll_list, test_list, _, _ = zip(*read_table(trial_score_file))
    enroll_list = sorted(list(set(enroll_list)))  # remove overlap and sort
    test_list = sorted(list(set(test_list)))
    enroll_emb, enroll_utt2idx = split_embedding(enroll_list, eval_emb_scp,
                                                 mean_vec)
    test_emb, test_utt2idx = split_embedding(test_list, eval_emb_scp, mean_vec)

    cohort_list, _ = zip(*read_table(cohort_emb_scp))
    cohort_emb, _ = split_embedding(cohort_list, cohort_emb_scp, mean_vec)

    logging.info("computing normed score ...")
    if score_norm_method == "asnorm":
        top_n = top_n
    elif score_norm_method == "snorm":
        top_n = cohort_emb.shape[0]
    else:
        raise ValueError(score_norm_method)
    enroll_mean, enroll_std = get_mean_std(enroll_emb, cohort_emb, top_n)
    test_mean, test_std = get_mean_std(test_emb, cohort_emb, top_n)

    # score norm
    with open(trial_score_file, 'r', encoding='utf-8') as fin:
        with open(score_norm_file, 'w', encoding='utf-8') as fout:
            lines = fin.readlines()
            for line in tqdm(lines):
                line = line.strip().split()
                enroll_idx = enroll_utt2idx[line[0]]
                test_idx = test_utt2idx[line[1]]
                score = float(line[2])
                normed_score = 0.5 * (
                    (score - enroll_mean[enroll_idx]) / enroll_std[enroll_idx]
                    + (score - test_mean[test_idx]) / test_std[test_idx])
                # compute mag mean for score calibration
                enroll_mag = np.linalg.norm(enroll_emb[enroll_idx])
                test_mag = np.linalg.norm(test_emb[test_idx])
                fout.write(
                    '{} {} {:.5f} {} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                        line[0], line[1], normed_score, line[3], enroll_mag,
                        test_mag, enroll_mean[enroll_idx],
                        test_mean[test_idx]))


if __name__ == "__main__":
    fire.Fire(main)
