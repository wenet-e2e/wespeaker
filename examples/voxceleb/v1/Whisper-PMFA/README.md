## Results

* Setup: mel80, num_frms500, epoch8, ArcMargin, aug_prob0.6, speed_perturb (no spec_aug)

* Scoring: cosine (sub mean of vox1_dev), AS-Norm

* Metric: EER(%)

* 🔥 UPDATE 2024.08: We support Whisper based speaker verification framework Whisper-PMFA. Related papers:

    * [Whisper-PMFA: Partial Multi-Scale Feature Aggregation for Speaker Verification using Whisper Models ](https://arxiv.org/pdf/2408.15585)



| Model                                | AS-Norm | Params | vox1-O-clean |
| :----------------------------------- | ------- | ------ | :----------: |
| ECAPA_TDNN_GLOB_c512-ASTP-emb192     | ×       | 6.19M  |     2.23     |
|                                      | √       | 6.19M  |     2.00     |
| ResNet34-TSTP-emb256                 | ×       | 6.63M  |     1.99     |
|                                      | √       | 6.63M  |     1.88     |
| Whisper-PMFA                         | ×       | 478.7M |     1.62     |
|                                      | √       | 478.7M |   **1.42**   |
| Whisper-PMFA with LoRA (Coming soon) | √       | 10.9M  |     1.62     |

