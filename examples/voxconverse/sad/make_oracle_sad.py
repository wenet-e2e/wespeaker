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


def read_ref_rttm(rttm):
    utt_to_segments = OrderedDict()

    for line in open(rttm, 'r'):
        line = line.strip().split()
        utt, begin, duration = line[1], line[3], line[4]
        begin = float(begin)
        end = begin + float(duration)
        if utt not in utt_to_segments:
            utt_to_segments[utt] = [(begin, end)]
        else:
            utt_to_segments[utt].append((begin, end))

    for utt in utt_to_segments.keys():
        utt_to_segments[utt].sort()

    return utt_to_segments


def merge_segments(utt_to_segments, min_duration):
    utt_to_merged_segments = OrderedDict()

    for utt, segments in utt_to_segments.items():
        utt_to_merged_segments[utt] = []
        if len(segments) > 0:
            (begin, end) = segments[0]
            for (b, e) in segments[1:]:
                assert begin <= b
                if b <= end:
                    end = max(end, e)
                else:
                    utt_to_merged_segments[utt].append((begin, end))
                    begin, end = b, e
            utt_to_merged_segments[utt].append((begin, end))

    segments_line_spec = "{}-{:08d}-{:08d} {} {:.3f} {:.3f}"
    for utt, segments in utt_to_merged_segments.items():
        for (begin, end) in segments:
            if end - begin >= min_duration:
                print(segments_line_spec.format(
                    utt, int(begin * 1000), int(end * 1000), utt, begin, end))


if __name__ == '__main__':
    utt_to_segments = read_ref_rttm(sys.argv[1])
    merge_segments(utt_to_segments, float(sys.argv[2]))
