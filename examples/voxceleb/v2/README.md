## Results

* Setup: fbank80, num_frms200, ArcMargin, aug_prob0.6, speed_perturb (no spec_aug)
* Scoring: cosine (sub mean of vox2_dev)
* Metric: EER(%)

| Model | Params | AS-Norm(300) | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| XVEC-TSTP-emb512 | 4.61M | × | 1.962 | 1.918 | 3.389 |
|                  |       | √ | 1.835 | 1.822 | 3.110 |
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.19M | × | 1.149 | 1.248 | 2.313 |
|                                  |       | √ | 1.026 | 1.154 | 2.089 |
| ResNet34-TSTP-emb256 | 6.70M | × | 0.941 | 1.114 | 2.026 |
|                      |       | √ | **0.899** | **1.064** | **1.856** |

