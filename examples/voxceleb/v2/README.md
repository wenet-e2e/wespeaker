## Results

* Setup: fbank80, num_frms200, epoch150, ArcMargin, aug_prob0.6, speed_perturb (no spec_aug)
* Scoring: cosine (sub mean of vox2_dev), AS-Norm, [QMF](https://arxiv.org/pdf/2010.11255)
* Metric: EER(%)
* 🔥 UPDATE 2024.09.03: We support the SimAM_ResNet pretrained on VoxBlink2 and Finetuned on Voxceleb2!
* 🔥 UPDATE 2024.08.27: We support SSL models as the feature front-end, take a look at the WavLM recipe!
* UPDATE 2022.07.19: We apply the same setups as the winning system of CNSRC 2022 (see [cnceleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/cnceleb/v2) recipe for details), and obtain significant performance improvement.
    * LR scheduler warmup from 0
    * Remove one embedding layer in ResNet models
    * Add large margin fine-tuning strategy (LM)

| Model | Params | Flops | LM | AS-Norm | QMF | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------|:--:|:-------:|:---:|:------------:|:------------:|:------------:|
| XVEC-TSTP-emb512 | 4.61M | 0.53G | × | × | × | 1.989 | 1.950 | 3.412 |
|                  |       |       | × | √ | × | 1.834 | 1.846 | 3.124 |
|                  |       |       | √ | × | × | 1.749 | 1.721 | 2.944 |
|                  |       |       | √ | √ | × | 1.590 | 1.641 | 2.726 |
| ECAPA_TDNN_GLOB_c512-ASTP-emb192  | 6.19M | 1.04G | × | × | × | 1.069 | 1.209 | 2.310 |
|                                   |       |       | × | √ | × | 0.957 | 1.128 | 2.105 |
|                                   |       |       | √ | × | × | 0.878 | 1.072 | 2.007 |
|                                   |       |       | √ | √ | × | 0.782 | 1.005 | 1.824 |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | 14.65M | 2.65G | × | × | × | 0.856 | 1.072 | 2.059 |
|                                   |        |       | × | √ | × | 0.808 | 0.990 | 1.874 |
|                                   |        |       | √ | × | × | 0.798 | 0.993 | 1.883 |
|                                   |        |       | √ | √ | × | 0.728 | 0.929 | 1.721 |
|                                   |        |       | √ | √ | √ | 0.707 | 0.894 | 1.615 |
| ResNet34-TSTP-emb256 | 6.63M | 4.55G | × | × | × | 0.867 | 1.049 | 1.959 |
|                      |       |       | × | √ | × | 0.787 | 0.964 | 1.726 |
|                      |       |       | × | √ | √ | 0.718 | 0.911 | 1.606 |
|                      |       |       | √ | × | × | 0.797 | 0.937 | 1.695 |
|                      |       |       | √ | √ | × | 0.723 | 0.867 | 1.532 |
|                      |       |       | √ | √ | √ | 0.659 | 0.821 | 1.437 |
| ResNet221-TSTP-emb256 | 23.79M | 21.29G | × | × | × | 0.569 | 0.774 | 1.464 |
|                       |        |        | × | √ | × | 0.479 | 0.707 | 1.290 |
|                       |        |        | √ | × | × | 0.580 | 0.729 | 1.351 |
|                       |        |        | √ | √ | × | 0.505 | 0.676 | 1.213 |
| ResNet293-TSTP-emb256 | 28.62M | 28.10G | × | × | × | 0.595 | 0.756 | 1.433 |
|                       |        |        | × | √ | × | 0.537 | 0.701 | 1.276 |
|                       |        |        | √ | × | × | 0.532 | 0.707 | 1.311 |
|                       |        |        | √ | √ | × | 0.447 | 0.657 | 1.183 |
|                       |        |        | √ | √ | √ | **0.425** | **0.641** | **1.146** |
| RepVGG_TINY_A0       | 6.26M | 4.65G | × | × | × | 0.909 | 1.034 | 1.943 |
|                      |       |       | × | √ | × | 0.824 | 0.953 | 1.709 |
| CAM++                | 7.18M | 1.15G | × | × | × | 0.803 | 0.932 | 1.860 |
|                      |       |       | × | √ | × | 0.718 | 0.879 | 1.735 |
|                      |       |       | √ | x | × | 0.707 | 0.845 | 1.664 |
|                      |       |       | √ | √ | × | 0.659 | 0.803 | 1.569 |
| ERes2Net34_Base      | 7.88M | 3.43G | × | × | × | 0.914 | 1.065 | 1.986 |
|                      |       |       | × | √ | × | 0.803 | 0.976 | 1.787 |
|                      |       |       | √ | x | × | 0.824 | 0.968 | 1.776 |
|                      |       |       | √ | √ | × | 0.744 | 0.896 | 1.603 |
| Res2Net34_Base       | 4.68M | 1.77G | × | × | × | 1.351 | 1.347 | 2.478 |
|                      |       |       | × | √ | × | 1.234 | 1.232 | 2.162 |
| Gemini_DFResNet114   | 6.53M | 5.42G | × | × | × | 0.787 | 0.963 | 1.760 |
|                      |       |       | × | √ | × | 0.707 | 0.889 | 1.546 |
|                      |       |       | √ | x | × | 0.771 | 0.906 | 1.599 |
|                      |       |       | √ | √ | × | 0.638 | 0.839 | 1.427 |
| SimAM_ResNet34 (VoxBlink2 Pretrain)        | 25.2M |       | √ | x | × | 0.415 | 0.615 | 1.121 |
|                      |       |       | √ | √ | × | 0.372 | 0.581 | 1.049 |
|                      |       |       | √ | √ | √ | 0.372 | 0.559 | 0.997 |
| SimAM_ResNet100 (VoxBlink2 Pretrain)       | 50.2M |       | √ | x | × | 0.229 | 0.458 | 0.868 |
|                      |       |       | √ | √ | × | 0.207 | 0.424 | 0.804 |
|                      |       |       | √ | √ | √ | 0.202 | 0.421 | 0.795 |
| XI_VEC_ECAPA_TDNN_c512       | 5.9M | 1.04G      | x | x | × | 0.995 | 1.130 | 2.169 |
|                  |       |       | × | √ | × | 0.883 | 1.056 | 1.976 |
|                  |       |       | √ | × | × | 0.909 | 1.000 | 1.855 |
|                  |       |       | √ | √ | × | 0.787 | 0.930 | 1.693 |


## PLDA results
If you are interested in the PLDA scoring (which is inferior to the simple cosine scoring under the margin based setting), simply run:

```bash
local/score_plda.sh --stage 1 --stop-stage 3 --exp_dir exp_name
```

The results on ResNet34 (large margin, no asnorm) are:

| Scoring method | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:--------------:|:------------:|:------------:|:------------:|
|      PLDA      |    1.207     |    1.350     |    2.528     |


## WavLM results

* Pre-trained frontend: the [WavLM](https://arxiv.org/abs/2110.13900) Large model, multilayer features are used
* Speaker model: ECAPA_TDNN_GLOB_c512-ASTP-emb192
* Training strategy: Frozen => Joint ft => Joint lmft

```bash
bash run_wavlm.sh --stage 3 --stop_stage 9
```

| Training strategy | AS-Norm | QMF | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------------------|:-------:|:---:|:------------:|:------------:|:------------:|
| Frozen            | × | × | 0.595 | 0.719 | 1.501 |
|                   | √ | × | 0.548 | 0.656 | 1.355 |
|                   | √ | √ | 0.489 | 0.619 | 1.224 |
| Frozen => Joint ft | × | × | 0.542 | 0.635 | 1.355 |
|                    | √ | × | 0.521 | 0.594 | 1.237 |
|                    | √ | √ | 0.494 | 0.576 | 1.205 |
| Frozen => Joint ft => Joint lmft | × | × | 0.521 | 0.626 | 1.344 |
|                                  | √ | × | 0.495 | 0.588 | 1.247 |
|                                  | √ | √ | **0.415** | **0.551** | **1.118** |
