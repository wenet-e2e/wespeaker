# Copyright (c) 2023 Brno University of Technology
#                    Shuai Wang (wsstriving@gmail.com)
#
# Python implementation of Kaldi unsupervised PLDA adaptation
# ( https://github.com/kaldi-asr/kaldi/blob/master/src/ivector/plda.cc#L613 )
# by Daniel Povey.
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
    parser.add_argument('--adp_scp',
                        '-ad',
                        type=str,
                        required=True,
                        help='Data for unlabeled adaptation.')
    parser.add_argument('--across_class_scale',
                        '-as',
                        type=float,
                        help='Scaling factor for across class covariance.',
                        default=0.5)
    parser.add_argument('--within_class_scale',
                        '-ws',
                        type=float,
                        help='Scaling factor for withn class covariance.',
                        default=0.5)
    parser.add_argument('--mdl_org',
                        '-mo',
                        type=str,
                        required=True,
                        help='Original PLDA mdl.')
    parser.add_argument('--mdl_adp',
                        '-ma',
                        type=str,
                        required=True,
                        help='Adapted PLDA mdl.')
    parser.add_argument('--mdl_format',
                        '-mf',
                        type=str,
                        default='wespeaker',
                        help='Format of the model wespeaker/kaldi')

    args = parser.parse_args()

    kaldi_format = True if args.mdl_format == 'kaldi' else False
    plda = TwoCovPLDA.load_model(args.mdl_org, kaldi_format)
    adapt_plda = plda.adapt(args.adp_scp, args.across_class_scale,
                            args.within_class_scale)
    adapt_plda.save_model(args.mdl_adp)
