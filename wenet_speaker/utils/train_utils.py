#!/usr/bin/env python3
# Copyright (c) 2021 Hongji Wang

import torch
import numpy as np
import random


def set_mannul_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

