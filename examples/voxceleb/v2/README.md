## Results

* Setup: fbank80, num_frms200, epoch150, ArcMargin, aug_prob0.6, speed_perturb (no spec_aug)
* Scoring: cosine (sub mean of vox2_dev)
* Metric: EER(%)

| Model | Params | AS-Norm(300) | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| XVEC-TSTP-emb512 | 4.61M | × | 1.962 | 1.918 | 3.389 |
|                  |       | √ | 1.835 | 1.822 | 3.110 |
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.19M | × | 1.149 | 1.248 | 2.313 |
|                                  |       | √ | 1.026 | 1.154 | 2.089 |
| ResNet34-TSTP-emb256 | 6.70M | × | 0.941 | 1.114 | 2.026 |
|                      |       | √ | 0.899 | 1.064 | 1.856 |

<br/>

* 🔥 UPDATE 2022.7.19: We apply the same setups as the winning system of CNSRC 2022 (see [cnceleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/cnceleb/v2) recipe for details), and obtain significant performance improvement compared with our previous implementation.
    * LR scheduler warmup from 0
    * Remove one embedding layer in ResNet models
    * Add large margin fine-tuning strategy (LM)

| Model | Params | LM | AS-Norm | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:--:|:-------:|:------------:|:------------:|:------------:|
| ECAPA_TDNN_GLOB_c512-ASTP-emb192  | 6.19M | × | × | 1.069 | 1.209 | 2.310 |
|                                   |       | × | √ | 0.957 | 1.128 | 2.105 |
|                                   |       | √ | × | 0.878 | 1.072 | 2.007 |
|                                   |       | √ | √ | 0.782 | 1.005 | 1.824 |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | 14.65M | × | × | 0.856 | 1.072 | 2.059 |
|                                   |        | × | √ | 0.808 | 0.990 | 1.874 |
|                                   |        | √ | × | 0.798 | 0.993 | 1.883 |
|                                   |        | √ | √ | 0.728 | 0.929 | 1.721 |
| ResNet34-TSTP-emb256 | 6.63M | × | × | 0.867 | 1.049 | 1.959 |
|                      |       | × | √ | 0.787 | 0.964 | 1.726 |
|                      |       | √ | × | 0.797 | 0.937 | 1.695 |
|                      |       | √ | √ | **0.723** | **0.867** | **1.532** |
