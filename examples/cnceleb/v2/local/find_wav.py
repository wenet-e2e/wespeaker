#!usr/bin/env python3
# coding=utf-8
# Author: Chengdong Liang

import os
import argparse
from tqdm import tqdm


def find_all_wav(dirname, extension='.wav'):
    if dirname[-1] != os.sep:
        dirname += os.sep
    for root, _, filenames in tqdm(os.walk(dirname, followlinks=True)):
        wavfiles = [f for f in filenames if f.endswith(extension)]
        if len(wavfiles) > 0:
            for _wav in wavfiles:
                print(os.path.join(root, _wav))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='data',
                        help='dataset_dir')
    parser.add_argument('--extension',
                        type=str,
                        default='wav',
                        help='file extension name')
    args = parser.parse_args()

    find_all_wav(args.data_dir, args.extension)
