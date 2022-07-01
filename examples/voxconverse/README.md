## Results
* We suggest to run this recipe on a gpu-available machine, with onnxruntime-gpu supported.
* Dataset: voxconverse_dev that consists of 216 utterances
* Speaker model: ResNet34 model pretrained by wespeaker
* Speaker activity detection model: oracle SAD (from ground truth annotation) or system SAD (VAD model pretrained by silero, https://github.com/snakers4/silero-vad)
* Clustering method: spectral clustering
* Metric: DER = MISS + FALSE ALARM + SPEAKER CONFUSION (%)

| system | MISS | FA | SC | DER |
|:---|:---:|:---:|:---:|:---:
| This repo (with oracle SAD) | 2.3 | 0.0 | 3.6 | 5.9 |
| This repo (with system SAD) | 4.4 | 0.6 | 3.6 | 8.6 |
| [1] DIHARD 2019 baseline | 11.1 | 1.4 | 11.3 | 23.8 |
| [1] DIHARD 2019 baseline w/ SE | 9.3 | 1.3 | 9.7 | 20.2 |
| [1] (SyncNet ASD only) | 2.2 | 4.1 | 4.0 | 10.4 |
| [1] (AVSE ASD only) | 2.0 | 5.9 | 4.6 | 12.4 |
| [1] (proposed) | 2.4 | 2.3 | 3.0 | 7.7 |

[1] Spot the conversation: speaker diarisation in the wild, https://arxiv.org/pdf/2007.01216.pdf