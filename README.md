## Result
* ECAPA_TDNN_aug0.6: vox1_testO: 1.13%EER

## TODO list:
* speed perturb && specaug ==> testing
* local/prepare_data.sh
* python cos/plda scoring


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
