## Results for SRE16

* Setup: fbank40, num_frms200, epoch150, Softmax, aug_prob0.6
* Scoring: cosine & PLDA & PLDA Adaptation
* Metric: EER(%)

Without PLDA training data augmentation:
| Model | Params | Backend | Pooled | Tagalog | Cantonese |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| ResNet34-TSTP-emb256 | 6.63M | Cosine | 15.65 | 20.38 | 10.38 |
|                      |       | PLDA | 9.506 | 14.73 | 4.363 |
|                      |       | Adapt PLDA | 6.802 | 10.4 | 3.14 |

With PLDA training data augmentation:
| Model | Params | Backend | Pooled | Tagalog | Cantonese |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| ResNet34-TSTP-emb256 | 6.63M | Cosine | 15.65 | 20.38 | 10.38 |
|                      |       | PLDA | 9.476 | 14.81 | 4.28 |
|                      |       | Adapt PLDA | 6.727 | 10.32 | 3.161 |
