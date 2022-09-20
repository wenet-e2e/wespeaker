# Copyright (c)  2020  Xiaomi Corporation (author: Fangjun Kuang)
#                2022  Binbin Zhang(binbzha@qq.com)
#                2022  Chengdong Liang(liangchengdong@mail.nwpu.edu.cn)
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

import setuptools

def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


package_name = "wespeakerruntime"

setuptools.setup(
    name=package_name,
    version='1.0.0',
    author="Chengdong Liang",
    author_email="liangchengdongd@qq.com",
    package_dir={
        package_name: "py",
    },
    packages=[package_name],
    url="https://github.com/wenet-e2e/wespeaker",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    setup_requires=["tqdm"],
    install_requires=[
        'onnxruntime',
        'kaldiio',
        'torchaudio',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache licensed, as found in the LICENSE file",
    python_requires=">=3.6",
)
