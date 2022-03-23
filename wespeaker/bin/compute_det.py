#!/usr/bin/env python3
# coding=utf-8
# Author: Chengdong Liang

import os
import fire
import numpy as np

from wespeaker.utils.score_metrics import compute_pmiss_pfa_rbst, plot_det_curve


def main(scores_dir, *trails):

    for trail in trails:
        scoresfile = os.path.join(scores_dir, trail + '.score')
        det_path = os.path.join(scores_dir, trail + '.det.png')
        scores = []
        labels = []
        with open(scoresfile) as readlines:
            for line in readlines:
                tokens = line.strip().split()
                assert len(tokens) == 4
                scores.append(float(tokens[2]))
                labels.append(tokens[-1] == 'target')

        scores = np.hstack(scores)
        labels = np.hstack(labels)

        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        plot_det_curve(fnr, fpr, det_path)


if __name__ == '__main__':
    fire.Fire(main)
