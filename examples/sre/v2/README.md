## Results for SRE16

* Setup: fbank40, num_frms200, epoch150, Softmax, aug_prob0.6
* Scoring: cosine & PLDA & PLDA Adaptation
* Metric: EER(%)

Without PLDA training data augmentation:
| Model | Params | Backend | Pooled | Tagalog | Cantonese |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| ResNet34-TSTP-emb256 | 6.63M | Cosine | 15.4 | 19.82 | 10.39 |
|                      |       | PLDA | 9.36 | 14.26 | 4.513 |
|                      |       | Adapt PLDA | 6.608 | 10.01 | 2.974 |

With PLDA training data augmentation:
| Model | Params | Backend | Pooled | Tagalog | Cantonese |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| ResNet34-TSTP-emb256 | 6.63M | Cosine | 15.4 | 19.82 | 10.39 |
|                      |       | PLDA | 8.944 | 13.54 | 4.462 |
|                      |       | Adapt PLDA | 6.543 | 9.666 | 3.254 |

* 🔥 UPDATE 2023.07.14: Support the [NIST SRE16 recipe](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016), see [#177](https://github.com/wenet-e2e/wespeaker/pull/177).
