# Copyright (c) 2022 Xu Xiang
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
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
import argparse
import kaldiio
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

import onnxruntime as ort
from wespeaker.utils.utils import validate_path


def init_session(source, device):
    # Initialize ONNX session
    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device == "cuda":
        providers = ["CUDAExecutionProvider"]
    else:
        raise ValueError

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 1
    session = ort.InferenceSession(source,
                                   sess_options=opts,
                                   providers=providers)
    return session


def read_fbank(scp_file):
    fbank_dict = OrderedDict()

    for utt, fbank in kaldiio.load_scp_sequential(scp_file):
        fbank_dict[utt] = fbank
    return fbank_dict


def subsegment(fbank, seg_id, window_fs, period_fs, frame_shift):
    subsegs = []
    subseg_fbanks = []

    seg_begin, seg_end = seg_id.split('-')[-2:]
    seg_length = (int(seg_end) - int(seg_begin)) // frame_shift

    # We found that the num_frames + 2 equals to seg_length, which is caused
    # by the implementation of torchaudio.compliance.kaldi.fbank.
    # Thus, here seg_length is used to get the subsegs.
    num_frames, feat_dim = fbank.shape
    if seg_length <= window_fs:
        subseg = seg_id + "-{:08d}-{:08d}".format(0, seg_length)
        subseg_fbank = np.resize(fbank, (window_fs, feat_dim))

        subsegs.append(subseg)
        subseg_fbanks.append(subseg_fbank)
    else:
        max_subseg_begin = seg_length - window_fs + period_fs
        for subseg_begin in range(0, max_subseg_begin, period_fs):
            subseg_end = min(subseg_begin + window_fs, seg_length)
            subseg = seg_id + "-{:08d}-{:08d}".format(subseg_begin, subseg_end)
            subseg_fbank = np.resize(fbank[subseg_begin:subseg_end],
                                     (window_fs, feat_dim))

            subsegs.append(subseg)
            subseg_fbanks.append(subseg_fbank)

    return subsegs, subseg_fbanks


def extract_embeddings(fbanks, batch_size, session, subseg_cmn):
    fbanks_array = np.stack(fbanks)
    if subseg_cmn:
        fbanks_array = fbanks_array - np.mean(
            fbanks_array, axis=1, keepdims=True)

    embeddings = []
    for i in tqdm(range(0, fbanks_array.shape[0], batch_size)):
        batch_feats = fbanks_array[i:i + batch_size]
        batch_embs = session.run(input_feed={'feats': batch_feats},
                                 output_names=['embs'])[0].squeeze()

        embeddings.append(batch_embs)
    embeddings = np.vstack(embeddings)

    return embeddings


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--scp', required=True, help='wav scp')
    parser.add_argument('--ark-path',
                        required=True,
                        help='path to store embedding ark')
    parser.add_argument('--source', required=True, help='onnx model')
    parser.add_argument('--device',
                        default='cuda',
                        help='inference device type: cpu or cuda')
    parser.add_argument('--batch-size',
                        type=int,
                        default=96,
                        help='batch size for embedding extraction')
    parser.add_argument('--frame-shift',
                        type=int,
                        default=10,
                        help='frame shift in fbank extraction (ms)')
    parser.add_argument('--window-secs',
                        type=float,
                        default=1.50,
                        help='the window seconds in embedding extraction')
    parser.add_argument('--period-secs',
                        type=float,
                        default=0.75,
                        help='the shift seconds in embedding extraction')
    parser.add_argument('--subseg-cmn',
                        default=True,
                        type=lambda x: x.lower() == 'true',
                        help='do cmn after or before fbank sub-segmentation')
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # transform duration to frame number
    window_fs = int(args.window_secs * 1000) // args.frame_shift
    period_fs = int(args.period_secs * 1000) // args.frame_shift

    session = init_session(args.source, args.device)
    fbank_dict = read_fbank(args.scp)

    subsegs, subseg_fbanks = [], []
    for seg_id, fbank in fbank_dict.items():
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank, seg_id, window_fs,
                                                    period_fs,
                                                    args.frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    embeddings = extract_embeddings(subseg_fbanks, args.batch_size, session,
                                    args.subseg_cmn)

    validate_path(args.ark_path)
    emb_ark = os.path.abspath(args.ark_path)
    emb_scp = emb_ark[:-3] + "scp"

    with kaldiio.WriteHelper('ark,scp:' + emb_ark + "," + emb_scp) as writer:
        for i, subseg_id in enumerate(subsegs):
            writer(subseg_id, embeddings[i])


if __name__ == '__main__':
    main()
