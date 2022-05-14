# WeSpeaker

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

* [VoxCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2)
* CNCeleb

## Support List:

* Model (SOTA Models)
    - [x] [Standard X-vector](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf)
    - [x] [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
    - [x] [ECAPA_TDNN](https://arxiv.org/abs/2005.07143)
* Pooling Functions
    - [x] TAP(mean) / TSDP(std) / TSTP(mean+std)
    - [x] Attentive Statistics Pooling (ASTP)
    - [ ] [Learnable Dictionary Encoding (LDE)](https://arxiv.org/pdf/1804.00385.pdf)
* Criteria
    - [x] Softmax
    - [x] [Sphere (A-Softmax)](https://www.researchgate.net/publication/327389164)
    - [x] [Add_Margin (AM-Softmax)](https://arxiv.org/pdf/1801.05599.pdf)
    - [x] [Arc_Margin (AAM-Softmax)](https://arxiv.org/pdf/1801.07698v1.pdf)
* Scoring
    - [x] Cosine
    - [ ] PLDA
    - [ ] Score Normalization (AS-Norm)
* Metric
    - [x] EER
    - [x] minDCF
* Online Augmentation
    - [x] Noise && RIR
    - [x] Speed Perturb
    - [x] Specaug
* Literature
    - [x] [Awesome Speaker Papers](https://github.com/wenet-e2e/wespeaker/blob/master/speaker_recognition_papers.md)

## Discussion

For Chinese users, you can scan the QR code on the left to follow our offical account of `WeNet Community`.
We also created a WeChat group for better discussion and quicker response. Please scan the QR code on the right to join the chat group.
| <img src="https://github.com/wenet-e2e/wenet-contributors/blob/main/wenet_official.jpeg" width="250px"> | <img src="https://github.com/wenet-e2e/wenet-contributors/blob/main/wespeaker/wangshuai.jpg" width="250px"> |
| ---- | ---- |

## Looking for contributors

If you are interested to contribute, feel free to contact @wsstriving or @robin1001
