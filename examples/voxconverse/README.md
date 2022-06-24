## Results

* dataset: voxconverse dev consists of 216 utterances
* speaker model: resnet model trained on VoxCeleb2 (avg_model.onnx)
* speaker activity detection: oracle (from annoataion) or system (from 'silero' vad)
* clustering method: spectral clustering
* Metric: DER (%) (MISS/FALSE ALARM/SPEAKER CONFUSION)

| Model | oracle VAD | system VAD |
|:------|:------:|:------------:|
| resnet | 4.8/0.5/3.6 |2.3/0/3.9|
