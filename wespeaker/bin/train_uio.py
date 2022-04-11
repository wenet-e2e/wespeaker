#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang

import fire

from wespeaker.utils.utils import parse_config_or_kwargs
from wespeaker.dataset.udataset import Dataset


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """

    configs = parse_config_or_kwargs(config, **kwargs)
    dataset = Dataset(
        configs['train_list'],
        configs['spk2id'],
        configs['dataset_conf'],
        reverb_lmdb_file=configs.get('reverb_lmdb', None),
        noise_lmdb_file=configs.get('noise_lmdb', None),
    )
    for i, item in enumerate(dataset):
        print(item['key'])
        if i > 5:
            break


if __name__ == '__main__':
    fire.Fire(train)
