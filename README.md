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


## Support list:
* Model (SOTA models):
    - [x] [Standard X-vector](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf)
    - [x] [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
    - [x] [ECAPA_TDNN](https://arxiv.org/abs/2005.07143) [[Source codes](https://github.com/lawlict/ECAPA-TDNN)]
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
* Augmentation：
    - [x] rir+noise
    - [x] speed perturb
    - [x] specaug
* Literature 
    - [ ] Awesome speaker papers



## Results
* Setup: fbank80, aug_prob0.6, ArcMargin, SGD

| Model | rir+noise | speed perturb | spec aug  | TEST O    | TEST E   | TEST H     |
|-------|-----------|---------------|-----------|-----------|----------|------------|
| ECAPA_TDNN_emb192_channels512 | Yes   | No    | No    | 1.170%    | 1.221%    | 2.234%    |
| ECAPA_TDNN_emb192_channels512 | Yes   | Yes   | No    | 1.085%    | 1.205%    | 2.288%    |
| ResNet34_emb256 | Yes   | No    | No    | 1.000%    | 1.149%    | 2.094%    |
| ResNet34_emb256 | Yes   | Yes   | No    | **1.000%**    | **1.067%**    | **1.990%**    |
