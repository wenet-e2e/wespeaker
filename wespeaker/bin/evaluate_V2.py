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

# TODO: Move LANG labels somewhere else 

VOXLINGUA107_LANG="""ab af am ar as az ba be bg bn bo br bs ca ceb cs cy da de el en eo es et eu fa
fi fo fr gl gn gu gv haw ha hi hr ht hu hy ia id is it iw ja jw ka kk km kn ko
la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru
sa sco sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi war
yi yo zh""".split()

def evaluate(config="conf/config.yaml", **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs["model_path"]
    embed_ark = configs["embed_ark"]
    batch_size = configs.get("batch_size", 1)
    num_workers = configs.get("num_workers", 1)
    utt_chunk = configs.get("utt_chunk") # NOTE: This will be used to chunk the utterance into 40 seconds segments
    eval_dataset = configs.get("eval_dataset") # NOTE: This will be used to chunk the utterance into 40 seconds segments

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    # Get spk2id_dict with labels
    data_label = configs["data_label"]
    data_utt_spk_list = read_table(data_label)

    
    # TODO: Refactor this
    # Add missing language labels so the model's matches projection layer
    if isinstance(eval_dataset, str) and "voxlingua_dev" in eval_dataset.lower():
        current_lang = [ item[1] for item in data_utt_spk_list ]
        for lng in VOXLINGUA107_LANG:
            if lng not in current_lang:
                data_utt_spk_list.append(["-", lng])

    spk2id_dict = spk2id(data_utt_spk_list)



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

    # Utt chunk 
    test_conf["utt_chunk"] = utt_chunk
    print("WARN: Setting utt_chunk =", utt_chunk)

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

    correct = 0
    total = 0
    result = []
    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            for _, batch in (enumerate(dataloader)):
                utts = batch["key"]
                targets = batch["label"]
                features = batch["feat"]
                features = features.float().to(device)  # (B,T,F)
                
                # Forward through model
                outputs = model(features, targets.float().to(device))  # embed or (embed_a, embed_b)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu() # (B,F)

                correct += torch.sum(torch.argmax(embeds, -1) == targets)

                embeds = embeds.cpu().detach().numpy()  # (B,F)

                total += len(targets.ravel())

                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)
                    
                    

    embed_path = os.path.dirname(embed_ark)
    with open("{}/result.txt".format(embed_path), "w") as f:
        f.write("Correct:  {}\n".format(correct.item()))
        f.write("Total:    {}\n".format(total))
        f.write("Accuracy: {}\n".format((correct / total).item()))

    print("Correct:  ", correct.item())
    print("Total:    ", total)
    print("Accuracy: ", (correct / total).item())


if __name__ == "__main__":
    fire.Fire(evaluate)
