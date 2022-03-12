# WeSpeaker

WeSpeaker mainly focuses on speaker embedding learning, with application to the speaker verification task. We support
online feature extraction or loading pre-extracted features in kaldi-format.

## Installation && Run

* Create Conda env: pytorch version >= 1.10.0 is required !!!

``` sh
conda create -n wenet_speaker python=3.9
conda activate wenet_speaker
conda install pytorch=1.10.1 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

* Run voxceleb recipe

``` sh
cd examples/voxceleb/v2
bash run.sh --stage 2 --stop-stage 4
```

## Dataset Support

- [Voxceleb dataset](https://github.com/wsstriving/wenet-speaker/tree/master/examples/voxceleb/v2)

### Support list:

* Model (SOTA models):
    - [x] [Standard X-vector](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf)
    - [x] [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
    - [x] [ECAPA_TDNN](https://arxiv.org/abs/2005.07143)
* Pooling functions
    - [x] TAP(mean) / TSDP(std) / TSTP(mean+std)
    - [x] Attentive statistics pooling (ASTP)
    - [ ] [Learnable Dictionary Encoding (LDE)](https://arxiv.org/pdf/1804.00385.pdf)
* Criteria
    - [x] softmax
    - [x] sphere
    - [x] [add_margin (AM-softmax)](https://arxiv.org/pdf/1801.05599.pdf)
    - [x] [arc_margin (AAM-softmax)](https://arxiv.org/pdf/1801.07698v1.pdf)
* Scoring:
    - [x] cosine scoring
    - [ ] python plda scoring
    - [ ] score normalization (AS-Norm)
* Online Augmentation：
    - [x] rir+noise
    - [x] speed perturb
    - [x] specaug
* Literature
    - [x] Awesome speaker papers

### Looking for contributors

If you are interested to contribute, feel free to contact @wsstriving or @robin1001
