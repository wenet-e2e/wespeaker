#!/usr/bin/env python3
# coding=utf-8
# Author: Chengdong Liang

import os
import kaldiio
import fire
import numpy as np
from tqdm import tqdm
import logging

from wespeaker.utils.file_utils import read_lists


def get_mean_std(emb, cohort, top_n):
    emb = emb / np.sqrt(np.sum(emb**2, axis=1))[:, None]
    cohort = cohort / np.sqrt(np.sum(cohort**2, axis=1))[:, None]
    emb_cohort_score = np.matmul(emb, cohort.T)
    emb_cohort_score = np.sort(emb_cohort_score, axis=1)[:, ::-1]
    emb_cohort_score_topn = emb_cohort_score[:, :top_n]

    emb_mean = np.mean(emb_cohort_score_topn, axis=1)
    emb_std = np.std(emb_cohort_score_topn, axis=1)

    return emb_mean, emb_std


def split_embedding(utt_list_file, emb_scp, mean_vec):
    utt_list = read_lists(utt_list_file)
    embs = []
    utt2idx = {}
    for utt, emb in kaldiio.load_scp_sequential(emb_scp):
        emb = emb - mean_vec
        if utt in utt_list:
            embs.append(np.array(emb))
            utt2idx[utt] = len(embs) - 1
    return np.array(embs), utt2idx


def main(enroll_list_file, test_list_file, cohort_list_file, score_norm_method,
         top_n, trials_score_file, score_norm_file, cal_mean, mean_path,
         cohort_emb_scp, eval_emb_scp):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # get embedding
    if not cal_mean:
        print("Do not do mean normalization for evaluation embeddings.")
        mean_vec = 0.0
    else:
        assert os.path.exists(mean_path)
        mean_vec = np.load(mean_path)

    # get embedding
    logging.info('get embedding ...')

    enroll_emb, enroll_utt2idx = split_embedding(enroll_list_file,
                                                 eval_emb_scp, mean_vec)
    test_emb, test_utt2idx = split_embedding(test_list_file, eval_emb_scp,
                                             mean_vec)
    cohort_emb, _ = split_embedding(cohort_list_file, cohort_emb_scp, mean_vec)

    logging.info("computing normed score ...")
    if score_norm_method == "asnorm":
        top_n = top_n
    elif score_norm_method == "snorm":
        print(cohort_emb.shape[0])
        top_n = cohort_emb.shape[0]
    else:
        raise ValueError(score_norm_method)
    enroll_mean, enroll_std = get_mean_std(enroll_emb, cohort_emb, top_n)
    test_mean, test_std = get_mean_std(test_emb, cohort_emb, top_n)

    # score norm
    with open(trials_score_file, 'r', encoding='utf-8') as fin:
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
                fout.write('{} {} {:.5f} {}\n'.format(line[0], line[1],
                                                      normed_score, line[3]))
    logging.info("Over!")


if __name__ == "__main__":
    fire.Fire(main)
