# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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

import kaldiio
from collections import OrderedDict


def read_scp(scp_file):
    """read scp file (also support PIPE format)

    Args:
        scp_file (str): path to the scp file

    Returns:
        list: key_value_list
    """
    key_value_list = []
    with open(scp_file, "r", encoding='utf8') as fin:
        for line in fin:
            tokens = line.strip().split()
            key = tokens[0]
            value = " ".join(tokens[1:])
            key_value_list.append((key, value))
    return key_value_list


def read_scp_dict(scp_file):
    """read scp file with exactly 2 columns

    Args:
        scp_file (str): path to the scp file

    Returns:
        dict: utt_to_wav
    """
    utt_to_wav = OrderedDict()
    for line in open(scp_file, 'r'):
        utt, wav = line.strip().split()
        utt_to_wav[utt] = wav

    return utt_to_wav


def read_lists(list_file):
    """read list file with only 1 column

    Args:
        list_file (str): path to the list file

    Returns:
        list: lists
    """
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_table(table_file):
    """read table file with any columns

    Args:
        table_file (str): path to the table file

    Returns:
        list: table_list
    """
    table_list = []
    with open(table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            tokens = line.strip().split()
            table_list.append(tokens)
    return table_list


def read_rttm(rttm_file):
    """read rttm_file: ``rttm`` is an annotation format,
                  which is widely-used in the diar task.

    Args:
        rttm_file (str): path to the rttm file

    Returns:
        dict: utt_to_segments
    """
    utt_to_segments = OrderedDict()

    for line in open(rttm_file, 'r'):
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


def read_labels(labels_file, frame_shift=10):
    """read labels_file (2 columns)

    Args:
        labels_file (str): path to the labels file
        frame_shift (int, optional): frame shift in ms. Defaults to 10.

    Returns:
        dict: utt_to_subseg_labels
    """
    utt_to_subseg_labels = OrderedDict()
    for line in open(labels_file, 'r'):
        subseg, label = line.strip().split()
        utt, begin_ms, end_ms, begin_frames, end_frames = subseg.split('-')
        begin = (int(begin_ms) + int(begin_frames) * frame_shift) / 1000.0
        end = (int(begin_ms) + int(end_frames) * frame_shift) / 1000.0
        if utt not in utt_to_subseg_labels:
            utt_to_subseg_labels[utt] = [(begin, end, label)]
        else:
            utt_to_subseg_labels[utt].append((begin, end, label))
    return utt_to_subseg_labels


def read_segments(segments_file):
    """read segments file (4 columns)

    Args:
        segments_file (str): path to the segments file

    Returns:
        dict: utt_to_segments
    """
    utt_to_segments = OrderedDict()
    for line in open(segments_file, 'r'):
        seg, utt, begin, end = line.strip().split()
        begin, end = float(begin), float(end)
        if utt not in utt_to_segments:
            utt_to_segments[utt] = [(seg, begin, end)]
        else:
            utt_to_segments[utt].append((seg, begin, end))

    return utt_to_segments


def read_fbank(scp_file):
    """read fbank feats from scp file

    Args:
        scp_file (str): path to the scp file

    Returns:
        dict: fbank_dict
    """

    fbank_dict = OrderedDict()

    for utt, fbank in kaldiio.load_scp_sequential(scp_file):
        fbank_dict[utt] = fbank
    return fbank_dict
