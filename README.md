## TODO list:
* speed perturb && specaug ==> testing
* python cos/plda scoring.


## Installation && Run

* Create Conda env: pytorch version >= 1.10.0 is required !!!

``` sh
conda create -n wenet_speaker python=3.9
conda activate wenet_speaker
conda install pytorch=1.10.1 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

* Train model: see conf/config.yaml and train.sh

``` sh
./train.sh
```

* Average model: see average_model.sh

``` sh
./average_model.sh --exp_dir exp/ECAPA_TDNN_SMALL_GLOB_emb256-fbank80-vox2_dev-aug0.6-spFalse-saFalse-ArcMargin-SGD-epoch66
```

* Extract embedding: see extract_vox.sh and extract_embedding.sh

``` sh
./extract_vox.sh --exp_dir exp/ECAPA_TDNN_SMALL_GLOB_emb256-fbank80-vox2_dev-aug0.6-spFalse-saFalse-ArcMargin-SGD-epoch66
```
 
* Score in kaldi: see score_kaldi dir (Kaldi is needed)

``` sh
./kaldiPLDA_vox1.sh --exp ../exp/ECAPA_TDNN_SMALL_GLOB_emb256-fbank80-vox2_dev-aug0.6-spFalse-saFalse-ArcMargin-SGD-epoch66 --data ../data
./kaldiCos_vox1.sh --exp ../exp/ECAPA_TDNN_SMALL_GLOB_emb256-fbank80-vox2_dev-aug0.6-spFalse-saFalse-ArcMargin-SGD-epoch66
```
