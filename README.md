## Result
* Model: ECAPA_TDNN_SMALL_GLOB_emb256-fbank80-vox2_dev-aug0.6-ArcMargin-SGD-epoch66

| speed perturb | spec aug  | TEST O    | TEST E   | TEST H     |
|---------------|-----------|-----------|----------|------------|
| No    | No    | 1.138%    | 1.247%    | 2.264%    |
| Yes   | No    | 1.096%    | 1.117%    | 2.105%    |
| No    | Yes   | 1.122%    | 1.191%    | 2.166%    |
| Yes   | Yes   | 1.186%    | 1.144%    | 2.131%    |


## TODO list:
* speed perturb && specaug ==> testing
* local/prepare_data.sh
* python plda scoring
* test model init
* test add_margin, sphere project_type
* SAP pooling method (Shuai Wang)


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
bash run.sh --stage 2 --stop-stage 2
```
