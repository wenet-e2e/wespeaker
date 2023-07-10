## Results

* Setup: fbank80, num_frms200(short) 300(long), epoch150, CE, aug_prob1.0, no speed_perturb, no spec_aug
* Scoring: cosine (sub mean of vox2_dev)
* Metric: EER(%)

| Model | Params | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------------:|:------------:|:------------:|
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.19M | 3.016 | 3.093 | 5.538 |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | ???M | 2.627 | 2.665 | 4.644 |
| ResNet34-TSTP-emb256 | 6.63M | 3.170 | 3.324 | 5.821 |


* 🔥 UPDATE 2023.07: We support DINO based self-supervised speaker verification framework.
