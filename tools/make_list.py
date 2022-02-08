#!/usr/bin/env python3

# Copyright (c) 2021 Hongji Wang           

import argparse
import json


def spk2id(spk_list):
    spk2id_dict = {}
    spk_list.sort()
    for i, spk in enumerate(spk_list):
        spk2id_dict[spk] = i
    return spk2id_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('wavscp_file', help='wavscp file')
    parser.add_argument('utt2spk_file', help='utt2spk file')
    parser.add_argument('output_file', help='output data.list file')
    args = parser.parse_args()

    wav_table = {}
    with open(args.wavscp_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    spk_set = set()
    with open(args.utt2spk_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            spk_set.add(arr[1])
    spk_list = list(spk_set)
    spk2id_dict = spk2id(spk_list)

    with open(args.utt2spk_file, 'r', encoding='utf8') as fin, \
            open(args.output_file, 'w', encoding='utf8') as fout:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            utt = arr[0]
            spk = arr[1]
            spk_id = spk2id_dict[spk]
            assert utt in wav_table
            wav = wav_table[utt]
            line = dict(key=utt, txt=spk_id, wav=wav)
            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + '\n')
