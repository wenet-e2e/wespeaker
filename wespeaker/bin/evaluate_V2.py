# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

import copy
import os

import fire
import kaldiio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wespeaker.dataset.dataset_V2 import Dataset
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import spk2id

# Speaker net abstraction with modoel and projection
from wespeaker.bin.train_V2 import SpeakerNet
from wespeaker.models.projections import get_projection


def evaluate(config="conf/config.yaml", **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs["model_path"]
    embed_ark = configs["embed_ark"]
    batch_size = configs.get("batch_size", 1)
    num_workers = configs.get("num_workers", 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    # projection layer
    configs["projection_args"]["embed_dim"] = configs["model_args"]["embed_dim"]
    configs["projection_args"]["num_class"] = len(spk2id_dict)
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
    load_checkpoint(model, model_path)

    device = torch.device("cuda")
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

    data_label = configs["data_label"]
    data_utt_spk_list = read_table(data_label)
    spk2id_dict = spk2id(data_utt_spk_list)

    dataset = Dataset(
        configs["data_type"],
        configs["data_list"],
        test_conf,
        spk2id_dict=spk2id_dict,
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

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    # TODO: Extract only the "embeddings" for eval but also input the accuracy.
    # For short recordings do some "partitioning"

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            for _, batch in tqdm(enumerate(dataloader)):
                utts = batch["key"]
                targets = batch["label"]
                features = batch["feat"]
                features = features.float().to(device)  # (B,T,F)

                # Forward through model
                outputs = model(features)  # embed or (embed_a, embed_b)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu().detach().numpy()  # (B,F)

                correct += (torch.argmax(embeds, axis=0) == targets).sum()
                total += len(targets)

                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)

        correct = correct.item()
        total = total.item()
        print("Correct:  ", correct)
        print("Total:    ", total)
        print("Accuracy: ", correct / total)


if __name__ == "__main__":
    fire.Fire(evaluate)
