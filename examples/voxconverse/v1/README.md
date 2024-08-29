## Overview

* We suggest to run this recipe on a gpu-available machine, with onnxruntime-gpu supported.
* Dataset: Voxconverse2020 (dev: 216 utts)
* Speaker model: ResNet34 model pretrained by WeSpeaker
  * Refer to [voxceleb sv recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2)
  * [pretrained model path](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx)
* Speaker activity detection model:
  * oracle SAD (from ground truth annotation)
  * system SAD (VAD model pretrained by [silero-vad](https://github.com/snakers4/silero-vad), v3.1 is deprecated now)
* Clustering method: spectral clustering
* Metric: DER = MISS + FALSE ALARM + SPEAKER CONFUSION (%)

## Results

* Dev set

    | system | MISS | FA | SC | DER |
    |:---|:---:|:---:|:---:|:---:|
    | Ours (oracle SAD + spectral clustering) | 2.3 | 0.0 | 1.9 | 4.2 |
    | Ours (silero-vad v3.1 + spectral clustering) | 3.7 | 0.8 | 2.0 | 6.5 |
    | DIHARD 2019 baseline [^1] | 11.1 | 1.4 | 11.3 | 23.8 |
    | DIHARD 2019 baseline w/ SE [^1] | 9.3 | 1.3 | 9.7 | 20.2 |
    | (SyncNet ASD only) [^1] | 2.2 | 4.1 | 4.0 | 10.4 |
    | (AVSE ASD only) [^1] | 2.0 | 5.9 | 4.6 | 12.4 |
    | (proposed) [^1] | 2.4 | 2.3 | 3.0 | 7.7 |


[^1]: Spot the conversation: speaker diarisation in the wild, https://arxiv.org/pdf/2007.01216.pdf

## Update 09/2022 : GPU Clustering
* You can use diar/clusterer\_gpu.py to run GPU Clustering
* We use [cupy](https://cupy.dev/) and [cuML](https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering) to accelarate the clustering speed
* You have to install the above toolkits before inference
* Similar performances can be obtained from our experiments but with ~3X speech up
* Try the test function in diar/clusterer\_gpu.py to have more details
