# WeSpeaker

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.8%7C3.9-brightgreen)](https://github.com/wenet-e2e/wespeaker)

[**Roadmap**](ROADMAP.md)
| [**Awesome Papers**](docs/speaker_recognition_papers.md)
| [**Runtime (x86_gpu)**](https://github.com/wenet-e2e/wespeaker/tree/master/runtime/server/x86_gpu)
| [**Pretrained Models**](docs/pretrained.md)


WeSpeaker mainly focuses on speaker embedding learning, with application to the speaker verification task. We support
online feature extraction or loading pre-extracted features in kaldi-format.

## Installation

* Clone this repo
``` sh
git clone https://github.com/wenet-e2e/wespeaker.git
```

* Create conda env: pytorch version >= 1.10.0 is required !!!
``` sh
conda create -n wespeaker python=3.9
conda activate wespeaker
conda install pytorch=1.10.1 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Recipes

* [VoxCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2): Speaker Verification recipe on the [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
    * 🔥 UPDATE 2022.7.19: We apply the same setups as the CNCeleb recipe, and obtain SOTA performance considering the open-source systems
* [CNCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/cnceleb/v2): Speaker Verification recipe on the [CnCeleb dataset](http://cnceleb.org/)
    * 🔥 UPDATE 2022.7.12: We are migrating the winner system of CNSRC 2022 [report](https://aishell-cnsrc.oss-cn-hangzhou.aliyuncs.com/T082.pdf) [slides](https://aishell-cnsrc.oss-cn-hangzhou.aliyuncs.com/T082-ZhengyangChen.pdf)
    * 🔥 EER/mindcf reduction from 8.426%/0.487 to **6.492%/0.354** after large margin fine-tuning and AS-Norm
* [VoxConverse](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxconverse): 🔥 UPDATE 2022.7.2:  Diarization recipe on the [VoxConverse dataset](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)

## Support List:

* Model (SOTA Models)
    - [x] [Standard X-vector](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf)
    - [x] [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
    - [x] [ECAPA_TDNN](https://arxiv.org/abs/2005.07143)
* Pooling Functions
    - [x] TAP(mean) / TSDP(std) / TSTP(mean+std)
        - Comparison of mean/std pooling can be found in [shuai_iscslp](https://x-lance.sjtu.edu.cn/en/papers/2021/iscslp21_shuai_1_.pdf), [anna_arxiv](https://arxiv.org/pdf/2203.10300.pdf)
    - [x] Attentive Statistics Pooling (ASTP)
        - mainly for ECAPA_TDNN
* Criteria
    - [x] Softmax
    - [x] [Sphere (A-Softmax)](https://www.researchgate.net/publication/327389164)
    - [x] [Add_Margin (AM-Softmax)](https://arxiv.org/pdf/1801.05599.pdf)
    - [x] [Arc_Margin (AAM-Softmax)](https://arxiv.org/pdf/1801.07698v1.pdf)
* Scoring
    - [x] Cosine
    - [ ] PLDA
    - [x] Score Normalization (AS-Norm)
* Metric
    - [x] EER
    - [x] minDCF
* Online Augmentation
    - [x] Noise && RIR
    - [x] Speed Perturb
    - [x] SpecAug
* Literature
    - [x] [Awesome Speaker Papers](https://github.com/wenet-e2e/wespeaker/blob/master/speaker_recognition_papers.md)

## Discussion

For Chinese users, you can scan the QR code on the left to follow our offical account of `WeNet Community`.
We also created a WeChat group for better discussion and quicker response. Please scan the QR code on the right to join the chat group.
| <img src="https://github.com/wenet-e2e/wenet-contributors/blob/main/wenet_official.jpeg" width="250px"> | <img src="https://github.com/wenet-e2e/wenet-contributors/blob/main/wespeaker/wangshuai.jpg" width="250px"> |
| ---- | ---- |

## Looking for contributors

If you are interested to contribute, feel free to contact @wsstriving or @robin1001
