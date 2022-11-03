# Copyright (c) 2022 Shuai Wang (wsstriving@gmail.com)
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

import argparse
import os
from wespeaker.utils.plda.two_cov_plda import TwoCovPLDA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',
                        type=str,
                        default='2cov',
                        help='which type of plda to use')
    parser.add_argument('--enroll_scp_path', type=str)
    parser.add_argument('--test_scp_path', type=str)
    parser.add_argument('--utt2spk', type=str)
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--trial', type=str)
    args = parser.parse_args()

    if args.type == '2cov':
        model_path = os.path.join(args.exp_dir, '2cov.plda')
        score_path = os.path.join(args.exp_dir, 'scores',
                                  os.path.basename(args.trial) + '.pldascore')
        plda = TwoCovPLDA.load_model(model_path)
        plda.eval_sv(args.enroll_scp_path, args.utt2spk, args.test_scp_path,
                     args.trial, score_path)
