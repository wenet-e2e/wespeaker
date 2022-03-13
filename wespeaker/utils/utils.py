#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang

import os
import logging
import torch
import numpy as np
import random
import yaml


def genlogger(outdir, fname):
    formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(
            level=logging.DEBUG,
            format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pyobj, f")
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_config_or_kwargs(config_file, **kwargs): 
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    help_str = "Valid Parameters are:\n"
    help_str += "\n".join(list(yaml_config.keys()))
    # passed kwargs will override yaml config
    #for key in kwargs.keys():
        #assert key in yaml_config, "Parameter {} invalid!\n".format(key) + help_str
    return dict(yaml_config, **kwargs)


def validate_path(dir_name):
    """ Create the directory if it doesn't exist
    :param dir_name
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name != ''):
        os.makedirs(dir_name)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def spk2id(utt_spk_list):
    spk_set=set()
    for utt_spk in utt_spk_list:
        spk_set.add(utt_spk[1])
    spk_list=list(spk_set)

    spk2id_dict = {}
    spk_list.sort()
    for i, spk in enumerate(spk_list):
        spk2id_dict[spk] = i
    return spk2id_dict

