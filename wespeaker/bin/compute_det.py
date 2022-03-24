#!/usr/bin/env python3
# coding=utf-8
# Author: Chengdong Liang

import fire
import numpy as np

from wespeaker.utils.score_metrics import compute_pmiss_pfa_rbst, plot_det_curve


def compute_det(scores_file, det_file):
    scores = []
    labels = []

    with open(scores_file) as readlines:
        for line in readlines:
            tokens = line.strip().split()
            assert len(tokens) == 4
            scores.append(float(tokens[2]))
            labels.append(tokens[-1] == 'target')

    scores = np.hstack(scores)
    labels = np.hstack(labels)

    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    plot_det_curve(fnr, fpr, det_file)
    print("DET curve saved in {}".format(det_file))


def main(*scores_files):
    for scores_file in scores_files:
        det_file = scores_file[:-6] + ".det.png"
        compute_det(scores_file, det_file)


if __name__ == '__main__':
    fire.Fire(main)
