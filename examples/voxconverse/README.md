## Results

* Dataset: voxconverse dev consists of 216 utterances
* Speaker model: resnet model trained on VoxCeleb2 (avg_model.onnx)
* Speaker activity detection: oracle (from annoataion) or system (from 'silero' vad)
* Clustering method: spectral clustering
* Metric: DER = MISS + FALSE ALARM + SPEAKER CONFUSION (%)

| system | MISS | FA | SC | DER |
|:---|:---:|:---:|:---:|:---:|
| Ours (with oracle sad) | 2.3 | 0.0 | 3.9 | 6.2 |
| Ours (with system sad) | 4.8 | 0.5 | 3.6 | 8.9 |
| [1] DIHARD 2019 baseline | 11.1 | 1.4 | 11.3 | 23.8 | 
| [1] DIHARD 2019 baseline w/ SE | 9.3 | 1.3 | 9.7 | 20.2 |
| [1] (SyncNet ASD only) | 2.2 | 4.1 | 4.0 | 10.4 |
| [1] (AVSE ASD only) | 2.0 | 5.9 | 4.6 | 12.4 |
| [1] (proposed) | 2.4 | 2.3 | 3.0 | 7.7 | 

[1] Spot the conversation: speaker diarisation in the wild, https://arxiv.org/pdf/2007.01216.pdf
