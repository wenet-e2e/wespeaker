# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
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
from pathlib import Path

import fire
import kaldiio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def calculate_mean_from_kaldi_vec(scp_path):
    vec_num = 0
    mean_vec = None

    for _, vec in kaldiio.load_scp_sequential(scp_path):
        if mean_vec is None:
            mean_vec = np.zeros_like(vec)
        mean_vec += vec
        vec_num += 1

    return mean_vec / vec_num


def trials_cosine_score(eval_scp_path='',
                        store_dir='',
                        mean_vec=None,
                        trials=()):
    if mean_vec is None or not os.path.exists(mean_vec):
        mean_vec = 0.0
    else:
        mean_vec = np.load(mean_vec)

    # each embedding may be accessed multiple times, here we pre-load them
    # into the memory
    emb_dict = {}
    for utt, emb in kaldiio.load_scp_sequential(eval_scp_path):
        emb = emb - mean_vec
        emb_dict[utt] = emb

    for trial in trials:
        store_path = os.path.join(store_dir,
                                  os.path.basename(trial) + '.score')
        with open(trial, 'r') as trial_r, open(store_path, 'w') as w_f:
            lines = trial_r.readlines()
            for line in tqdm(lines,
                             desc='scoring trial {}'.format(
                                 os.path.basename(trial))):
                segs = line.strip().split()
                emb1, emb2 = emb_dict[segs[0]], emb_dict[segs[1]]
                cos_score = cosine_similarity(emb1.reshape(1, -1),
                                              emb2.reshape(1, -1))[0][0]

                if len(segs) == 3:  # enroll_name test_name target/nontarget
                    w_f.write('{} {} {:.5f} {}\n'.format(
                        segs[0], segs[1], cos_score, segs[2]))
                else:  # enroll_name test_name
                    w_f.write('{} {} {:.5f}\n'.format(segs[0], segs[1],
                                                      cos_score))


def main(exp_dir, eval_scp_path, cal_mean, cal_mean_dir, *trials):

    print(cal_mean)
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


if __name__ == "__main__":
    fire.Fire(main)
