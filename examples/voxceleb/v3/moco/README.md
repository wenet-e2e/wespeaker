## Results

* Setup: fbank80, num_frms200, epoch150, InfoNCE, aug_prob1.0, speed_perturb, no spec_aug
* Scoring: cosine (sub mean of vox2_dev)
* Metric: EER(%)

| Model | Params | Methods | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.19M | MoCo | 8.709 | 9.287 | 14.756 |

* 🔥 UPDATE 2023.07: We support the MoCo based self-supervised speaker verification. Related papers:
    * [Momentum Contrast for Unsupervised Visual Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)
    * [Self-supervised Text-independent Speaker Verification using Prototypical Momentum Contrastive Learning](https://arxiv.org/abs/2012.07178)