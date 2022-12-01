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
| ResNet34-TSTP-emb256 | 6.63M | × | 0.941 | 1.114 | 2.026 |
|                      |       | √ | 0.899 | 1.064 | 1.856 |

<br/>


* We apply the same setups as the winning system of CNSRC 2022 (see [cnceleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/cnceleb/v2) recipe for details), and obtain significant performance improvement compared with our previous implementation.
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
|                      |       | √ | √ | 0.723 | 0.867 | 1.532 |
| ResNet221-TSTP-emb256 | 23.86M | × | × | 0.569 | 0.774 | 1.464 |
|                      |       | × | √ | 0.479 | 0.707 | 1.290 |
|                      |       | √ | × | 0.580 | 0.729 | 1.351 |
|                      |       | √ | √ | 0.505 | 0.676 | 1.213 |
| ResNet293-TSTP-emb256 | 28.69M | × | × | 0.595 | 0.756 | 1.433 |
|                      |       | × | √ | 0.537 | 0.701 | 1.276 |
|                      |       | √ | × | 0.532 | 0.707 | 1.311 |
|                      |       | √ | √ | **0.447** | **0.657** | **1.183** |
| RepVGG_TINY_A0       | 6.26M | × | × | 0.909 | 1.034 | 1.943 |
|                      |       | × | √ | 0.824 | 0.953 | 1.709 |


* 🔥 UPDATE 2022.11.30: We support arc_margin_intertopk_subcenter loss function and Multi-query Multi-head Attentive Statistics Pooling, and obtain better performance especially on hard trials [VoxSRC](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition2021.html).
    * See [#103](https://github.com/wenet-e2e/wespeaker/pull/103).


## PLDA results
If you are interested in the PLDA scoring (which is inferior to the simple cosine scoring under the margin based setting), simply run:

```bash
local/score_plda.sh --stage 1 --stop-stage 3 --exp_dir exp_name
```

The results on ResNet293 (large margin, no asnorm) are:

|Scoring method| vox1-O-clean | vox1-E-clean | vox1-H-clean |
| :---:|:------------:|:------------:|:------------:|
|cosine| 0.532 | 0.707 | 1.311 |
|plda | 0.744 | 0.794 | 1.374|
