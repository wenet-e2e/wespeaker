# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
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

import os

import fire
import kaldiio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wespeaker.dataset.dataset_deprecated import FeatList_LableDict_Dataset
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.file_utils import read_scp
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path


def extract(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    data_scp = configs['data_scp']
    embed_ark = configs['embed_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)
    raw_wav = configs.get('raw_wav', True)
    feat_dim = configs['feature_args'].get('feat_dim', 80)
    num_frms = configs['feature_args'].get('num_frms', 200)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    model = get_speaker_model(configs['model'])(**configs['model_args'])
    load_checkpoint(model, model_path)
    device = torch.device("cuda")
    model.to(device).eval()

    # prepare dataset and dataloader
    data_list = read_scp(data_scp)
    dataset = FeatList_LableDict_Dataset(data_list,
                                         utt2spkid_dict={},
                                         whole_utt=(batch_size == 1),
                                         raw_wav=raw_wav,
                                         feat_dim=feat_dim,
                                         num_frms=num_frms)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            t_bar = tqdm(ncols=100,
                         total=len(dataloader),
                         desc='extract_embed: ')
            for i, (utts, feats, _) in enumerate(dataloader):
                t_bar.update()

                feats = feats.float().to(device)  # (B,T,F)
                # Forward through model
                outputs = model(feats)  # embed or (embed_a, embed_b)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu().detach().numpy()  # (B,F)

                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)

            t_bar.close()


if __name__ == '__main__':
    fire.Fire(extract)
