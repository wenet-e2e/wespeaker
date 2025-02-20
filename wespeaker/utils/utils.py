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

import logging
import os
import random

import numpy as np
import torch
import yaml
from distutils.util import strtobool
from pathlib import Path
import shutil



def get_logger(outdir, fname):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(level=logging.DEBUG,
                        format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pyobj, f")
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def setup_logger(rank, exp_dir, MAX_NUM_LOG_FILES: int = 100):
    model_dir = os.path.join(exp_dir, "models")
    file_name = "train.log"
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        for i in range(MAX_NUM_LOG_FILES - 1, -1, -1):
            if i == 0:
                p = Path(os.path.join(exp_dir, file_name))
                pn = p.parent / (p.stem + ".1" + p.suffix)
            else:
                _p = Path(os.path.join(exp_dir, file_name))
                p = _p.parent / (_p.stem + f".{i}" + _p.suffix)
                pn = _p.parent / (_p.stem + f".{i + 1}" + _p.suffix)

            if p.exists():
                if i == MAX_NUM_LOG_FILES - 1:
                    p.unlink()
                else:
                    shutil.move(p, pn)
    return get_logger(exp_dir, file_name)

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
    # for key in kwargs.keys():
    #    assert key in yaml_config, "Parameter {} invalid!\n".format(key)
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

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def spk2id(utt_spk_list):
    _, spk_list = zip(*utt_spk_list)
    spk_list = sorted(set(spk_list))  # remove overlap and sort

    spk2id_dict = {}
    for i, spk in enumerate(spk_list):
        spk2id_dict[spk] = i
    return spk2id_dict
