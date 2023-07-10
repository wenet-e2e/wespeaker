## Results

* Setup: fbank80, num_frms200, epoch150, InfoNCE, aug_prob1.0, speed_perturb, no spec_aug
* Scoring: cosine (sub mean of vox2_dev)
* Metric: EER(%)

| Model | Params | Methods | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.19M | SimCLR | 8.523 | 9.417 | 14.907 |

* 🔥 UPDATE 2023.07: We support the SimCLR based self-supervised speaker verification. Related Papers:
    * [A Simple Framework for Contrastive Learning of Visual Representations](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf)
    * [Contrastive Self-Supervised Learning for Text-Independent Speaker Verification](https://ieeexplore.ieee.org/document/9413351)