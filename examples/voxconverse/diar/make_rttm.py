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

# Note that the frame shift value defaults to 10 milliseconds
def read_labels(labels, frame_shift=10):
    subseg_to_labels = []
    for line in open(labels, 'r'):
        subseg, label = line.strip().split()
        utt, begin_ms, end_ms, begin_frames, end_frames = subseg.split('-')
        begin = (int(begin_ms) + int(begin_frames) * frame_shift) / 1000.0
        end = (int(begin_ms) + int(end_frames) * frame_shift) / 1000.0
        subseg_to_labels.append((utt, begin, end, label))
    return subseg_to_labels


def merge_segments(subseg_to_labels):
    if len(subseg_to_labels) == 0:
        return []

    merged_segment_to_labels = []
    (utt, begin, end, label) = subseg_to_labels[0]
    for (u, b, e, l) in subseg_to_labels[1:]:
        if b <= end and l == label:
            end = e
        elif b > end:
            merged_segment_to_labels.append((utt, begin, end, label))
            utt, begin, end, label = u, b, e, l
        elif b <= end and l != label:
            pivot = (b + end) / 2.0
            merged_segment_to_labels.append((utt, begin, pivot, label))
            utt, begin, end, label = u, pivot, e, l
        else:
            raise ValueError
    merged_segment_to_labels.append((utt, begin, e, label))
    return merged_segment_to_labels


if __name__ == '__main__':
    subseg_to_labels = read_labels(sys.argv[1])
    merged_segment_to_labels = merge_segments(subseg_to_labels)
    DEFAULT_CHANNEL = 1
    if len(sys.argv) == 3:
        channel = int(sys.argv[2])
    else:
        channel = DEFAULT_CHANNEL
    rttm_line_spec = "SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>"
    for (utt, begin, end, label) in merged_segment_to_labels:
        print(rttm_line_spec.format(utt, channel, begin, end - begin, label))
