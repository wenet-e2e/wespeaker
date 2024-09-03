# Copyright (c) 2022  Mddct(hamddct@gmail.com)
#               2023  Binbin Zhang(binbzha@qq.com)
#               2024  Shuai Wang(wsstriving@gmail.com)
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
import tarfile
import zipfile
from urllib.request import urlretrieve

import tqdm


def download(url: str, dest: str, only_child=True):
    """download from url to dest"""
    assert os.path.exists(dest)
    print("Downloading {} to {}".format(url, dest))

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
    name = url.split("?")[0].split("/")[-1]
    file_path = os.path.join(dest, name)
    with tqdm.tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=(name)
    ) as t:
        urlretrieve(
            url, filename=file_path, reporthook=progress_hook(t), data=None
        )
        t.total = t.n

    if name.endswith((".tar.gz", ".tar")):
        with tarfile.open(file_path) as f:
            if not only_child:
                f.extractall(dest)
            else:
                for tarinfo in f:
                    if "/" not in tarinfo.name:
                        continue
                    name = os.path.basename(tarinfo.name)
                    fileobj = f.extractfile(tarinfo)
                    with open(os.path.join(dest, name), "wb") as writer:
                        writer.write(fileobj.read())

    elif name.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            if not only_child:
                zip_ref.extractall(dest)
            else:
                for member in zip_ref.namelist():
                    member_path = os.path.relpath(
                        member, start=os.path.commonpath(zip_ref.namelist())
                    )
                    if "/" not in member_path:
                        continue
                    name = os.path.basename(member_path)
                    with zip_ref.open(member_path) as source, open(
                        os.path.join(dest, name), "wb"
                    ) as target:
                        target.write(source.read())


class Hub(object):
    Assets = {
        "chinese": "cnceleb_resnet34.tar.gz",
        "english": "voxceleb_resnet221_LM.tar.gz",
        "campplus": "campplus_cn_common_200k.tar.gz",
        "eres2net": "eres2net_cn_commom_200k.tar.gz",
        "vblinkp": "voxblink2_samresnet34.zip",
        "vblinkf": "voxblink2_samresnet34_ft.zip",
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_model(lang: str) -> str:
        if lang not in Hub.Assets.keys():
            print("ERROR: Unsupported lang {} !!!".format(lang))
            sys.exit(1)
        model = Hub.Assets[lang]
        model_dir = os.path.join(Path.home(), ".wespeaker", lang)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if set(["avg_model.pt", "config.yaml"]).issubset(
            set(os.listdir(model_dir))
        ):
            return model_dir
        else:
            response = requests.get(
                "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"  # noqa
            )
            model_info = next(
                data
                for data in response.json()["Data"]
                if data["Key"] == model
            )
            model_url = model_info["Url"]
            download(model_url, model_dir)
            return model_dir
