# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2024 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2024 Bing Han (hanbing97@sjtu.edu.cn)
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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from wespeaker.utils.file_utils import read_table


def gather_calibration_factors(wav_dur_scp, max_dur, score_norm_file,
                               calibration_factor_file, drop_duration=False):
    if not drop_duration:
        wav_idx, dur_list = zip(*read_table(wav_dur_scp))
        wavidx2dur = {
            idx: min(float(dur), max_dur)
            for idx, dur in zip(wav_idx, dur_list)
        }

    def reorder_values(value_1, value_2):
        max_value = max(value_1, value_2)
        min_value = min(value_1, value_2)
        return "{:.4f} {:.4f} {:.4f} {:.4f}".format(min_value, max_value,
                                                    max_value - min_value,
                                                    max_value / min_value)

    # read factor from asnorm results
    assert os.path.exists(
        score_norm_file), "score norm file ({}) does not exist !!!".format(
            score_norm_file)

    with open(score_norm_file, 'r', encoding='utf-8') as fin:
        with open(calibration_factor_file, 'w', encoding='utf-8') as fout:
            lines = fin.readlines()
            for line in tqdm(lines):
                line = line.strip().split()
                idx1, idx2 = line[0], line[1]
                if drop_duration:
                    dur_str = ""
                else:
                    dur_str = reorder_values(wavidx2dur[idx1], wavidx2dur[idx2])
                mag_str = reorder_values(float(line[4]), float(line[5]))
                cohort_mean_str = reorder_values(float(line[6]),
                                                 float(line[7]))
                fout.write('{} {} {} {} {} {} {}\n'.format(
                    line[0], line[1], line[3], line[2], dur_str, mag_str,
                    cohort_mean_str))


class LinearModel(nn.Module):

    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.constant_(self.linear.weight, 1.0 / input_dim)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        out = self.linear(x)
        return out


def cllr(target_llrs, nontarget_llrs):
    """
    Calculate the CLLR of the scores
    """

    def negative_log_sigmoid(lodds):
        """-log(sigmoid(log_odds))"""
        return torch.log1p(torch.exp(-lodds))

    return 0.5 * (torch.mean(negative_log_sigmoid(target_llrs)) + torch.mean(
        negative_log_sigmoid(-nontarget_llrs))) / np.log(2)


def train_calibration_model(calibration_factor_file, save_model_path):
    max_epochs = 50
    target_llrs_list = []
    nontarget_llrs_list = []
    with open(calibration_factor_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            if line[2] == "tgt" or line[2] == "target":
                target_llrs_list.append([float(v) for v in line[3:]])
            else:
                nontarget_llrs_list.append([float(v) for v in line[3:]])

    # build training set
    target_llrs = torch.tensor(target_llrs_list, dtype=torch.float64)
    nontarget_llrs = torch.tensor(nontarget_llrs_list, dtype=torch.float64)
    start_cllr = cllr(target_llrs, nontarget_llrs)

    # create model
    model = LinearModel(target_llrs.shape[-1])
    model.double()
    criterion = cllr

    # build optimizer
    optimizer = optim.LBFGS(model.parameters(), lr=0.01)

    best_loss = 1000000.0
    for i in range(max_epochs):

        def closure():
            optimizer.zero_grad()
            new_nontarget_llrs = model(nontarget_llrs)
            new_target_llrs = model(target_llrs)
            loss = criterion(new_target_llrs, new_nontarget_llrs)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if (best_loss - loss < 1e-4):
            break
        else:
            if loss < best_loss:
                best_loss = loss

    torch.save(model.state_dict(), save_model_path)


def infer_calibration(calibration_factor_file, save_model_path,
                      calibration_score_file):
    llrs_list = []
    with open(calibration_factor_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            llrs_list.append([float(v) for v in line[3:]])

    llrs = torch.tensor(llrs_list, dtype=torch.float64)

    model = LinearModel(llrs.shape[-1])
    model.load_state_dict(torch.load(save_model_path))
    model.eval()
    model.double()
    outputs = model(llrs)

    with open(calibration_score_file, "w", encoding='utf-8') as fout:
        for i, s in enumerate(lines):
            line = lines[i].strip().split()
            score = outputs[i].item()
            fout.write('{} {} {} {}\n'.format(line[0], line[1], score,
                                              line[2]))


if __name__ == "__main__":
    fire.Fire()
