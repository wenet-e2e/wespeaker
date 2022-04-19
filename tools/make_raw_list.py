#!/usr/bin/env python3

# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com)
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
import logging
import os
import json


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('speaker_file', help='speaker file')
    parser.add_argument('raw_list', help='output raw list file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = os.path.splitext(arr[0])[0]
            assert len(arr) == 2
            wav_table[key] = arr[1]

    data = []
    with open(args.speaker_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = os.path.splitext(arr[0])[0]
            txt = arr[1]
            assert key in wav_table
            wav = wav_table[key]
            data.append((key, txt, wav))

    with open(args.raw_list, 'w', encoding='utf8') as fout:
        for key, speaker, wav in data:
            line = dict(key=key, speaker=speaker, wav=wav)
            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + '\n')


if __name__ == '__main__':
    main()