# Copyright (c) 2022  Mddct(hamddct@gmail.com)
#               2023  Binbin Zhang(binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import requests
import sys
from pathlib import Path
from urllib.request import urlretrieve

import tqdm


def download(url: str, dest: str):
    """ download from url to dest
    """
    print('Downloading {} to {}'.format(url, dest))

    def progress_hook(t):
        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            if tsize not in (None, -1):
                t.total = tsize
            displayed = t.update((b - last_b[0]) * bsize)
            last_b[0] = b
            return displayed

        return update_to

    # *.tar.gz
    name = url.split('?')[0].split('/')[-1]
    with tqdm.tqdm(unit='B',
                   unit_scale=True,
                   unit_divisor=1024,
                   miniters=1,
                   desc=(name)) as t:
        urlretrieve(url, filename=dest, reporthook=progress_hook(t), data=None)
        t.total = t.n


class Hub(object):
    Assets = {
        "chinese": "cnceleb_resnet34.onnx",
        "english": "voxceleb_resnet221_LM.onnx",
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_model(lang: str) -> str:
        if lang not in Hub.Assets.keys():
            print('ERROR: Unsupported lang {} !!!'.format(lang))
            sys.exit(1)
        model = Hub.Assets[lang]
        model_path = os.path.join(Path.home(), ".wespeaker", model)
        if not os.path.exists(model_path):
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            response = requests.get(
                "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"  # noqa
            )
            model_info = next(data for data in response.json()["Data"]
                              if data["Key"] == model)
            model_url = model_info['Url']
            download(model_url, model_path)
        return model_path
