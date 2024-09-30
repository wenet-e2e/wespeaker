#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description: Extracts single embedding from the wav file.
# Author: Ondřej Odehnal <xodehn09@vutbr.cz>
# =============================================================================
"""Extracts single embedding from the wav file."""
# =============================================================================
# Imports
# =============================================================================
import copy
import json

import fire
import torch
import numpy as np
from scipy.special import softmax
from torch.utils.data import DataLoader
import torch.nn as nn

from wespeaker.dataset.dataset_V2 import Dataset
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path
from wespeaker.models.projections import get_projection

from pathlib import Path


## Podoblasti
SUBREGIONS = {
    "1-1": "Severovýchodočeská",
    "1-2": "Středočeská",
    "1-3": "Jihozápadočeská",
    "1-4": "Českomoravská",
    "2-1": "Jižní",
    "2-2": "Západní",
    "2-3": "Východní",
    "2-4": "Centrální",
    "3-1": "Slovácko ",
    "3-2": "Zlínsko",
    "3-3": "Valašsko",
    "4-1": "Slezskomoravská",
    "4-2": "Slezskopolská",
}

CZECH_DIALECT_SUBCLASSES_NUMBER = len(SUBREGIONS)


class SpeakerNet(nn.Module):
    def __init__(self, model, projecNet):
        super(SpeakerNet, self).__init__()
        self.speaker_extractor = model
        self.projection = projecNet

    def forward(self, x, y):
        x = self.speaker_extractor(x)
        x = self.projection(x, y)
        return x


# TODO: How to handle flags with fire lib?
def extract(
    input_wav_file: str,
    output_embedding_path: str,
    model_path: str,
    overwrite: bool = True,
    config: str = "conf/config.yaml",
    device_type: str = "cpu",
    **kwargs,
):
    """Extracts single embedding from the wav file.

    Args:
        input_wav_file (str): Input wav file.
        output_embedding_path (str): Output embedding path for the extracted embedding of the input wav file by the model. Defaults to "out/".
        model_path (str): Model path.
        overwrite (bool, optional): Overwrite the output. Defaults to True.
        config (str, optional): Configuration for the model. Defaults to "conf/config.yaml".
        device_type (str, optional): Device type for torch, either "cpu" or "gpu". Defaults to "cpu".
    """

    assert Path(
        input_wav_file
    ).exists(), f"File wav_file {input_wav_file} does not exist!"
    assert (
        not Path(output_embedding_path).exists() or overwrite
    ), f"File output_embedding_path {output_embedding_path} already exists!"
    assert Path(config).exists(), f"File config {config} does not exist!"
    assert device_type in ["cpu", "cuda"], f"Invalid device_type {device_type}!"

    # parse configs first and set the pre-defined device and data_type for embedding extraction
    configs = parse_config_or_kwargs(config, **kwargs)
    configs["model_args"]["device"] = device_type
    configs["data_type"] = "raw"

    configs["model_args"]["model_path"] = model_path

    batch_size = configs.get("batch_size", 1)
    num_workers = configs.get("num_workers", 1)
    # TODO: Consider using the utt_chunk parameter
    utt_chunk = configs.get(
        "utt_chunk", False
    )  # NOTE: This will be used to chunk the utterance into 40 seconds segments if set to True

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    model = get_speaker_model(configs["model"])(**configs["model_args"])
    ############################################################
    # projection layer
    ############################################################
    configs["projection_args"]["embed_dim"] = configs["model_args"]["embed_dim"]
    configs["projection_args"]["num_class"] = CZECH_DIALECT_SUBCLASSES_NUMBER
    configs["projection_args"]["do_lm"] = configs.get("do_lm", False)
    if configs["data_type"] != "feat" and configs["dataset_args"]["speed_perturb"]:
        # diff speed is regarded as diff spk
        configs["projection_args"]["num_class"] *= 3
    projection = get_projection(configs["projection_args"])

    # Load model with projection layer
    model = SpeakerNet(
        get_speaker_model(configs["model"])(**configs["model_args"]),
        projection,
    )
    ############################################################

    load_checkpoint(model, model_path)
    device = torch.device(device_type)
    model.to(device).eval()

    # test_configs
    test_conf = copy.deepcopy(configs["dataset_args"])
    test_conf["speed_perturb"] = False
    if "fbank_args" in test_conf:
        test_conf["fbank_args"]["dither"] = 0.0
    elif "mfcc_args" in test_conf:
        test_conf["mfcc_args"]["dither"] = 0.0
    test_conf["spec_aug"] = False
    test_conf["shuffle"] = False
    test_conf["aug_prob"] = configs.get("aug_prob", 0.0)
    test_conf["filter"] = False

    # Utt chunk
    test_conf["utt_chunk"] = utt_chunk
    print("WARN: Setting utt_chunk =", utt_chunk)

    # NOTE: We are processing single utterances but we are using the Dataset to handle the processing
    input_wav_json = json.dumps(
        {"key": Path(input_wav_file).name, "wav": str(input_wav_file), "spk": "-"}
    )
    dataset = Dataset(
        configs["data_type"],
        [input_wav_json],  # TODO: Change this to List[str]
        test_conf,
        spk2id_dict={},
        whole_utt=(batch_size == 1),
        reverb_lmdb_file=configs.get("reverb_data", None),
        noise_lmdb_file=configs.get("noise_data", None),
        repeat_dataset=False,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4,
    )

    # Process the utterance
    with torch.no_grad():

        # NOTE: we have only one utterance but we still use the dataloader to handle the processing
        dataloader_iterator = iter(enumerate(dataloader))
        i, batch = next(dataloader_iterator)

        utts = batch["key"]
        print(f"[{i}] Proccesing utts: {utts[0]}", flush=True)

        features = batch["feat"]
        features = features.float().to(device)  # (B,T,F)
        # Forward through model
        dummy_target = torch.zeros(features.size(0), dtype=torch.float)
        outputs = model(
            features, dummy_target.float().to(device)
        )  # embed or (embed_a, embed_b)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        embeds = embeds.cpu().detach().numpy()  # (B,F)

        # NOTE: We take only the last embedding. We are processing only one utterance.
        embed = embeds[i]

    # Saving the extracted embed
    if not Path(output_embedding_path).exists() or overwrite:
        np.savetxt(fname=output_embedding_path, X=embed, delimiter=",")
        print("Embedding saved to:", output_embedding_path)

    subregion_key, subregion_name = list(SUBREGIONS.items())[int(embed.argmax())]
    print(
        f"Predicted subregion {subregion_key} : {subregion_name} with probability {softmax(embed).max()*100:.2f} %"
    )


if __name__ == "__main__":
    fire.Fire(extract)
