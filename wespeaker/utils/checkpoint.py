#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang

import torch


def load_checkpoint(model: torch.nn.Module, path: str):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)


def save_checkpoint(model: torch.nn.Module, path: str):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
