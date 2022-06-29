# Copyright (c) 2022 Xu Xiang
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


import sys
from collections import OrderedDict


def read_labels(labels, frame_shift=10):
    utt_to_subseg_labels = OrderedDict()
    for line in open(labels, 'r'):
        subseg, label = line.strip().split()
        utt, begin_ms, end_ms, begin_frames, end_frames = subseg.split('-')
        begin = (int(begin_ms) + int(begin_frames) * frame_shift) / 1000.0
        end = (int(begin_ms) + int(end_frames) * frame_shift) / 1000.0
        if utt not in utt_to_subseg_labels:
            utt_to_subseg_labels[utt] = [(begin, end, label)]
        else:
            utt_to_subseg_labels[utt].append((begin, end, label))
    return utt_to_subseg_labels


def merge_segments(utt_to_subseg_labels):
    merged_segment_to_labels = []

    for utt, subseg_to_labels in utt_to_subseg_labels.items():
        if len(subseg_to_labels) == 0:
            continue

        (begin, end, label) = subseg_to_labels[0]
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


def main():
    subseg_to_labels = read_labels(sys.argv[1])

    # VoxConverse and DIHARD dataset: defaults to 1
    DEFAULT_CHANNEL = 1
    if len(sys.argv) == 3:
        channel = int(sys.argv[2])
    else:
        channel = DEFAULT_CHANNEL

    merged_segment_to_labels = merge_segments(subseg_to_labels)

    rttm_line_spec = "SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>"
    for (utt, begin, end, label) in merged_segment_to_labels:
        print(rttm_line_spec.format(utt, channel, begin, end - begin, label))


if __name__ == '__main__':
    main()
