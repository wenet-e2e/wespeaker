# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
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

import os

import fire
import numpy as np

from wespeaker.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer,
                                           compute_c_norm)


def compute_metrics(scores_file, p_target=0.01, c_miss=1, c_fa=1):
    scores = []
    labels = []

    with open(scores_file) as readlines:
        for line in readlines:
            tokens = line.strip().split()
            # assert len(tokens) == 4
            scores.append(float(tokens[2]))
            labels.append(tokens[3] == 'target')

    scores = np.hstack(scores)
    labels = np.hstack(labels)

    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = compute_eer(fnr, fpr, scores)

    min_dcf = compute_c_norm(fnr,
                             fpr,
                             p_target=p_target,
                             c_miss=c_miss,
                             c_fa=c_fa)
    print("---- {} -----".format(os.path.basename(scores_file)))
    print("EER = {0:.3f}".format(100 * eer))
    print("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.3f}".format(
        p_target, c_miss, c_fa, min_dcf))


def main(p_target=0.01, c_miss=1, c_fa=1, *scores_files):
    for scores_file in scores_files:
        compute_metrics(scores_file, p_target, c_miss, c_fa)


if __name__ == "__main__":
    fire.Fire(main)
