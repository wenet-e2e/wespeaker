#!/usr/bin/env python3
# coding=utf-8
# Author: Zhengyang Chen

import os
import kaldiio
from tqdm import tqdm
import numpy as np
from pathlib import Path
import fire
from sklearn.metrics.pairwise import cosine_similarity
from wespeaker.utils.score_metrics import compute_pmiss_pfa_rbst, compute_eer, compute_c_norm


def calculate_mean_from_kaldi_vec(scp_path):
    vec_num = 0
    mean_vec = None

    for _, vec in kaldiio.load_scp_sequential(scp_path):
        if mean_vec is None:
            mean_vec = np.zeros_like(vec)
        mean_vec += vec
        vec_num += 1

    return mean_vec / vec_num


def compute_metrics(scoresfile, p_target=0.01, c_miss=1, c_fa=1):
    scores = []
    labels = []

    exist_label = True
    with open(scoresfile) as readlines:
        for line in readlines:
            tokens = line.strip().split()

            if len(tokens) == 3:
                exist_label = False
                break

            scores.append(float(tokens[2]))
            labels.append(tokens[-1] == 'target')

    if exist_label:
        scores = np.hstack(scores)
        labels = np.hstack(labels)

        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer, thres = compute_eer(fnr, fpr, scores)

        min_dcf = compute_c_norm(fnr, fpr, p_target=p_target, c_miss=c_miss, c_fa=c_fa)
        print("---- {} -----".format(os.path.basename(scoresfile)))
        print("EER = {0:.3f}".format(100 * eer))
        print("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.3f}".format(p_target, c_miss, c_fa, min_dcf))


def trials_cosine_score(eval_scp_path='', store_dir='', mean_vec=None, trials=()):
    if mean_vec is None or not os.path.exists(mean_vec):
        mean_vec = 0.0
    else:
        mean_vec = np.load(mean_vec)

    # each embedding may be accessed multiple times, here we pre-load them into the memory
    emb_dict = {}
    for utt, emb in kaldiio.load_scp_sequential(eval_scp_path):
        emb = emb - mean_vec
        emb_dict[utt] = emb

    for trial in trials:
        store_path = os.path.join(store_dir, os.path.basename(trial) + '.score')
        with open(trial, 'r') as trial_r, open(store_path, 'w') as w_f:
            lines = trial_r.readlines()
            for line in tqdm(lines, desc='scoring trial {}'.format(os.path.basename(trial))):
                segs = line.strip().split()
                emb1, emb2 = emb_dict[segs[0]], emb_dict[segs[1]]
                cos_score = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

                if len(segs) == 3:  # enroll_name test_name target/nontarget
                    w_f.write('{} {} {:.5f} {}\n'.format(segs[0], segs[1], cos_score, segs[2]))
                else:  # enroll_name test_name
                    w_f.write('{} {} {:.5f}\n'.format(segs[0], segs[1], cos_score))


def main(exp_dir, eval_scp_path, cal_mean, cal_mean_dir, p_target=0.01, c_miss=1, c_fa=1, *trials):
    if not cal_mean:
        print("Do not do mean normalization for evaluation embeddings.")
        mean_vec_path = None
    else:
        scp_path = os.path.join(cal_mean_dir, 'xvector.scp')
        print("Calculate mean statistics from {}.".format(scp_path))
        mean_vec = calculate_mean_from_kaldi_vec(scp_path)
        mean_vec_path = os.path.join(cal_mean_dir, 'mean_vec.npy')
        np.save(mean_vec_path, mean_vec)

    # scoring trials
    store_score_dir = os.path.join(exp_dir, 'scores')
    Path(store_score_dir).mkdir(parents=True, exist_ok=True)
    trials_cosine_score(eval_scp_path, store_score_dir, mean_vec_path, trials)

    # compute evaluation metric
    for trial in trials:
        score_path = os.path.join(store_score_dir, os.path.basename(trial) + '.score')
        compute_metrics(score_path, p_target, c_miss, c_fa)


if __name__ == "__main__":
    fire.Fire(main)
