## Results

* Setup: fbank80, num_frms200(short) 300(long), epoch150, CE, aug_prob1.0, no speed_perturb, no spec_aug
* Scoring: cosine (sub mean of vox2_dev)
* Metric: EER(%)

| Model | Params | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------------:|:------------:|:------------:|
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.19M | 3.016 | 3.093 | 5.538 |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | 14.65M | 2.627 | 2.665 | 4.644 |
| ResNet34-TSTP-emb256 | 6.63M | 3.170 | 3.324 | 5.821 |


* 🔥 UPDATE 2023.07: We support DINO based self-supervised speaker verification framework. Related papers:
    * [A comprehensive study on self-supervised distillation for speaker representation learning](https://arxiv.org/pdf/2210.15936.pdf)
    * [Emerging properties in self-supervised vision transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)
    * [Self-supervised speaker verification using dynamic loss-gate and label correction](https://arxiv.org/pdf/2208.01928.pdf)