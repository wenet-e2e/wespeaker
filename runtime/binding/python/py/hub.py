# Copyright (c) 2022  Mddct(hamddct@gmail.com)
#               2022  Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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
from pathlib import Path
from urllib.request import urlretrieve

import tqdm


def download(url: str, dest: str, model_path: str):
    """ download from url to dest
    """
    assert os.path.exists(dest)
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
    name = url.split("/")[-1]
    with tqdm.tqdm(unit='B',
                   unit_scale=True,
                   unit_divisor=1024,
                   miniters=1,
                   desc=(name)) as t:
        urlretrieve(url,
                    filename=model_path,
                    reporthook=progress_hook(t),
                    data=None)
        t.total = t.n


class Hub(object):
    """Hub for wespeaker pretrain onnx model
    """
    # TODO(Mddct): make assets class to support other language
    Assets = {
        # cnceleb
        "chs":
        "https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx",
        # voxceleb
        "en":
        "https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx"
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_model_by_lang(lang: str) -> str:
        assert lang in Hub.Assets.keys()
        # NOTE(Chengdong Liang): model_dir structure
        # Path.Home()/.wespeaker
        # - chs
        #    - model.onnx
        # - en
        #    - model.onnx
        model_url = Hub.Assets[lang]
        model_dir = os.path.join(Path.home(), ".wespeaker", lang)
        model_path = os.path.join(model_dir, 'model.onnx')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # TODO(Mddct): model metadata
        if os.path.exists(model_path):
            return model_path
        download(model_url, model_dir, model_path)
        return model_path
