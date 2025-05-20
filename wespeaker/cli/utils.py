# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#                    Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t',
                        '--task',
                        choices=[
                            'embedding',
                            'embedding_kaldi',
                            'similarity',
                            'diarization',
                            'diarization_list',
                        ],
                        default='embedding',
                        help='task type')
    parser.add_argument('-l',
                        '--language',
                        choices=[
                            'chinese',
                            'english',
                        ],
                        default='chinese',
                        help='language type')
    parser.add_argument(
        '--campplus',
        action='store_true',
        help='whether to use the damo/speech_campplus_sv_zh-cn_16k-common model'
    )
    parser.add_argument(
        '--eres2net',
        action='store_true',
        help='whether to use the damo/speech_eres2net_sv_zh-cn_16k-common model'
    )
    parser.add_argument(
        '--vblinkp',
        action='store_true',
        help='whether to use the samresnet34 model pretrained on voxblink2'
    )
    parser.add_argument(
        '--vblinkf',
        action='store_true',
        help="whether to use the samresnet34 model pretrained on voxblink2 and"
             "fintuned on voxceleb2"
    )
    parser.add_argument('-p',
                        '--pretrain',
                        type=str,
                        default="",
                        help='model directory')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help="device type (most commonly cpu or cuda,"
                             "but also potentially mps, xpu, xla or meta)"
                             "and optional device ordinal for the device type.")
    parser.add_argument('--audio_file', help='audio file')
    parser.add_argument('--audio_file2',
                        help='audio file2, specifically for similarity task')
    parser.add_argument('--wav_scp',
                        help='path to wav.scp, for extract and saving '
                        'kaldi-stype embeddings')
    parser.add_argument('--resample_rate',
                        type=int,
                        default=16000,
                        help='resampling rate')
    parser.add_argument('--vad',
                        action='store_true',
                        help='whether to do VAD or not')
    parser.add_argument('--output_file',
                        default=None,
                        help='output file to save speaker embedding '
                        'or save diarization result')
    # diarization params
    parser.add_argument('--diar_min_duration',
                        type=float,
                        default=0.255,
                        help='VAD min duration')
    parser.add_argument('--diar_window_secs',
                        type=float,
                        default=1.5,
                        help='the window seconds in embedding extraction')
    parser.add_argument('--diar_period_secs',
                        type=float,
                        default=0.75,
                        help='the shift seconds in embedding extraction')
    parser.add_argument('--diar_frame_shift',
                        type=int,
                        default=10,
                        help='frame shift in fbank extraction (ms)')
    parser.add_argument('--diar_emb_bs',
                        type=int,
                        default=32,
                        help='batch size for embedding extraction')
    parser.add_argument('--diar_subseg_cmn',
                        type=bool,
                        default=True,
                        help='do cmn after or before fbank sub-segmentation')
    args = parser.parse_args()
    return args
