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
    """
    Currently, we only support the two-cov version,
    more variants will be added in next release.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',
                        type=str,
                        default='2cov',
                        help='which type of plda to use, we only support '
                        'kaldi 2cov version currently')
    parser.add_argument('--scp_path',
                        type=str,
                        help='the plda training embedding.scp file')
    parser.add_argument('--utt2spk', type=str, help='utt2spk file')
    parser.add_argument('--indim',
                        type=int,
                        help='the dimension of input embeddings')
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--iter', type=int, default=5)
    args = parser.parse_args()

    if args.type == '2cov':
        plda = TwoCovPLDA(scp_file=args.scp_path,
                          utt2spk_file=args.utt2spk,
                          embed_dim=args.indim)
        plda.train(args.iter)
        model_path = os.path.join(args.exp_dir, 'plda')
        plda.save_model(model_path)
