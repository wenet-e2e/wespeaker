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

from wespeaker.utils.plda.two_cov_plda import TwoCovPLDA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',
                        type=str,
                        default='2cov',
                        help='which type of plda to use, 2cov|kaldi')
    parser.add_argument('--enroll_scp_path',
                        type=str,
                        help='enroll embeddings')
    parser.add_argument('--indomain_scp_path',
                        type=str,
                        help='embeddings to compute meanvec')
    parser.add_argument('--test_scp_path', type=str, help='test embeddings')
    parser.add_argument('--utt2spk',
                        type=str,
                        help='utt2spk for the enroll speakers')
    parser.add_argument('--model_path', type=str, help='pretrained plda path')
    parser.add_argument('--score_path',
                        type=str,
                        help='score file to write to')
    parser.add_argument('--trial', type=str, help='trial file to score upon')
    parser.add_argument('--multisession_avg', default=False, action="store_true",
                        help='Whether to score multisession by average instead '
                        'of by-the-book. Default False.')

    args = parser.parse_args()

    kaldi_format = True if args.type == 'kaldi' else False
    plda = TwoCovPLDA.load_model(args.model_path, kaldi_format)
    plda.eval_sv(args.enroll_scp_path, args.utt2spk, args.test_scp_path,
                 args.trial, args.score_path, args.multisession_avg,
                 args.indomain_scp_path)
