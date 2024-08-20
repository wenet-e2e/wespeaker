## Overview

* We suggest to run this recipe on a gpu-available machine, with onnxruntime-gpu supported.
* Dataset: voxconverse_dev that consists of 216 utterances
* Speaker model: ResNet34 model pretrained by wespeaker
  * Refer to [voxceleb sv recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2)
  * [pretrained model path](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx)
* Speaker activity detection model: oracle SAD (from ground truth annotation) or system SAD (VAD model pretrained by silero, https://github.com/snakers4/silero-vad)
* Clustering method: umap dimensionality reduction + hdbscan clustering
* Metric: DER = MISS + FALSE ALARM + SPEAKER CONFUSION (%)

## Results

* Dev set

    | system | MISS | FA | SC | DER |
    |:---|:---:|:---:|:---:|:---:|
    | This repo (with oracle SAD) | 2.3 | 0.0 | 1.3 | 3.6 |
    | This repo (with system SAD) | 3.4 | 0.6 | 1.4 | 5.4 |
    | DIHARD 2019 baseline [^1] | 11.1 | 1.4 | 11.3 | 23.8 |
    | DIHARD 2019 baseline w/ SE [^1] | 9.3 | 1.3 | 9.7 | 20.2 |
    | (SyncNet ASD only) [^1] | 2.2 | 4.1 | 4.0 | 10.4 |
    | (AVSE ASD only) [^1] | 2.0 | 5.9 | 4.6 | 12.4 |
    | (proposed) [^1] | 2.4 | 2.3 | 3.0 | 7.7 |

* Test set

    | system | MISS | FA | SC | DER |
    |:---|:---:|:---:|:---:|:---:|
    | This repo (with oracle SAD) | 1.6 | 0.0 | 1.9 | 3.5 |
    | This repo (with system SAD) | 3.8 | 1.7 | 1.8 | 7.4 |


[^1]: Spot the conversation: speaker diarisation in the wild, https://arxiv.org/pdf/2007.01216.pdf
