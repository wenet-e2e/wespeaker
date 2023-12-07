# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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
import torch
import fire
import yaml

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.models.repvgg import repvgg_model_convert
from wespeaker.utils.utils import parse_config_or_kwargs


def convert(config='conf/config.yaml', **kwargs):
    configs = parse_config_or_kwargs(config, **kwargs)
    speaker_model = get_speaker_model(
        configs['model'])(**configs['model_args'])
    configs['model_args']['deploy'] = True
    # save new configs for testing and deploying
    # NOTE: 'deploy': true
    saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)

    if os.path.isfile(configs['load']):
        print("==> Loading checkpoint '{}'".format(configs['load']))
        checkpoint = torch.load(configs['load'])
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        speaker_model.load_state_dict(ckpt, strict=False)
    else:
        print('no checkpoint')
    repvgg_model_convert(speaker_model, save_path=configs['save'])
    print("==> Saving convert model to '{}'".format(configs['save']))


if __name__ == '__main__':
    fire.Fire(convert)
