# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

import fire
import numpy as np

from wespeaker.utils.score_metrics import compute_pmiss_pfa_rbst, plot_det_curve


def compute_det(scores_file, det_file):
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
    plot_det_curve(fnr, fpr, det_file)
    print("DET curve saved in {}".format(det_file))


def main(*scores_files):
    for scores_file in scores_files:
        det_file = scores_file + ".det.png"
        compute_det(scores_file, det_file)


if __name__ == '__main__':
    fire.Fire(main)
