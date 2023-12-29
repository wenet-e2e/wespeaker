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

import numpy as np


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


def merge_segments(utt_to_subseg_labels):
    merged_segment_to_labels = []

    for utt, subseg_to_labels in utt_to_subseg_labels.items():
        if len(subseg_to_labels) == 0:
            continue

        (begin, end, label) = subseg_to_labels[0]
        e = end  # when there is only one subseg, we assign end to e
        for (b, e, la) in subseg_to_labels[1:]:
            if b <= end and la == label:
                end = e
            elif b > end:
                merged_segment_to_labels.append((utt, begin, end, label))
                begin, end, label = b, e, la
            elif b <= end and la != label:
                pivot = (b + end) / 2.0
                merged_segment_to_labels.append((utt, begin, pivot, label))
                begin, end, label = pivot, e, la
            else:
                raise ValueError
        merged_segment_to_labels.append((utt, begin, e, label))

    return merged_segment_to_labels


def process_seg_id(seg_id, frame_shift=10):
    begin_ms, end_ms, begin_frames, end_frames = seg_id.split('-')
    begin = (int(begin_ms) + int(begin_frames) * frame_shift) / 1000.0
    end = (int(begin_ms) + int(end_frames) * frame_shift) / 1000.0

    return begin, end
