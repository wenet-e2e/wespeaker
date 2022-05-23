# Introduction
In this project, we use models trained in [wespeaker](https://github.com/wenet-e2e/wespeaker) as an example to show how to convert speaker model to tensorrt and deploy them on [Triton Inference Server](https://github.com/triton-inference-server/server.git).

## Step 0. Train a model
Please follow wespeaker examples to train a model. After training, you should get several checkpoints under your `exp/xxx/models/` folder. We take [voxceleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2) as an example.

## Step 1. Export model
We'll first export our model to onnx and then convert our onnx model to tensorrt.
```
# go to your example
cd wespeaker/examples/voxceleb/v2
. ./path.sh
exp_dir=exp/resnet
python3 wespeaker/bin/export_onnx_gpu.py --config=${exp_dir}/config.yaml --checkpoint=${exp_dir}/models/avg_model.pt --output_model=${exp_dir}/models/avg_model.onnx

# If you want to minus the mean vector in the onnx model, you may simply add the --mean_vec to the .npy mean vector file.
python3 wespeaker/bin/export_onnx_gpu.py --config=${exp_dir}/config.yaml --checkpoint=exp/resnet/models/avg_model.pt --output_model=exp/resnet/models/avg_model.onnx --mean_vec=${exp_dir}/embeddings/vox2_dev/mean_vec.npy
```

Now let's convert our onnx model to tensorrt engine. We will deploy our model on Triton 22.03 therefore we here will use tensorrt 22.03 docker as an example to show how to convert the model. Please move your onnx model to the target platform/GPU you will deploy.

```
docker run --gpus '"device=0"' -it -v <the output onnx model directory>:/models nvcr.io/nvidia/tensorrt:22.03-py3
cd /models/
# shape=BxTxF  batchsize, sequence_length, feature_size
trtexec --saveEngine=b1_b128_s3000_fp16.trt  --onnx=/models/avg_model.onnx --minShapes=feats:1x200x80 --optShapes=feats:64x200x80 --maxShapes=feats:128x3000x80 --fp16
```
Here we get an engine which has maximum sequence length of 3000 and minimum length of 200. Since the frame stride is 10ms, 200 and 3000 corresponds to 2.02 seconds and 30.02 seconds respectively(kaldi feature extractor). Notice these numbers will differ and depend on your feature extractor parameters. Notice we've added `--fp16` and in pratice, we found this option will not affect the final accuracy and improve the perf at the same time.

You may set these numbers by your production requirements. If you only know the seconds of audio you will use and have no idea of how many frames it will generate, you may try the below script:
```python
import torchaudio.compliance.kaldi as kaldi
import torch
audio_dur_in_seconds = 2
feat_dim = 80  # please check config.yaml if you dont know
sample_rate = 16000

waveform = torch.ones(sample_rate * audio_dur_in_seconds).unsqueeze(0)
feat_tensor = kaldi.fbank(waveform,
                            num_mel_bins=feat_dim,
                            frame_shift=10,
                            frame_length=25,
                            energy_floor=0.0,
                            window_type='hamming',
                            htk_compat=True,
                            use_energy=False,
                            dither=1)
print(feat_tensor.shape) # (198, 80)
```
Then you may find `198` is the actual number of frames for audio of 2 seconds long.

That's it！We build an engine that can accept 2.02 to 30.02 seconds long audio. If your application can accept fixed audio segments, we suggest you to set the `minShapes`, `optShapes` and `maxShapes` to the same shape.

Now edit the config file under `model_repo/speaker_model/config.pbtxt` and replace `default_model_filename:xxx` with the name of your engine (e.g., `b1_b128_s3000_fp16.trt`) and put the engine under `model_repo/speaker_model/1/`.

And if you use other model settings or different model from ours (resnet34), for example, ecapa model, the embedding dim of which is 192, therefore, you should edit the `model_repo/speaker_model/config.pbtxt` and `model_repo/speaker/config.pbtxt` and set embedding dim to 192.

## Step 2. Build server and start server

Notice we use triton 22.03 in dockerfile. Be sure to use the triton that has the same version as your tensorrt.

Build server:
```
# server
docker build . -f Dockerfile/dockerfile.server -t wespeaker:latest --network host
```

```
docker run --gpus '"device=0"' -v $PWD/model_repo:/ws/model_repo --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti  wespeaker:latest
tritonserver --model-repository=/ws/model_repo
```
The port `8000` is for http request and `8001` for grpc request.

## Step 3. Build client and start client

Build client:

```
# client
docker build . -f Dockerfile/dockerfile.client -t wespeaker_client:latest --network host
```

```
docker run -it -v $PWD:/ws -v <data path>:<data path> --network=host wespeaker_client

# example command
cd /ws/client/
python3 client.py --url=<ip of the server>:8001 --wavscp=/raid/dgxsa/slyne/wespeaker/examples/voxceleb/v2/data/vox1/wav.scp --output_directory=<to put the generated embeddings>

# The output direcotry will be something like:
# xvector_000.ark xvextor_000.scp xvector_001.scp .....

```

## Step 4. Test score
After you extract the embeddings, you can now use the same way as wespeaker to test these embeddings. For example, you may test the extracted embeddings in wespeaker by:
```
cat embeddings/xvector_*.scp > embeddings/xvector.scp

config=conf/resnet.yaml
exp_dir=exp/resnet

mkdir -p embeddings/scores
trials_dir=data/vox1/trials
python -u wespeaker/bin/score.py \
    --exp_dir ${exp_dir} \
    --eval_scp_path /raid/dgxsa/slyne/wespeaker/runtime/server/x86_gpu/embeddings/xvector.scp \  # embeddings generated from our server
    --cal_mean True \
    --cal_mean_dir ${exp_dir}/embeddings/vox2_dev \
    --p_target 0.01 \
    --c_miss 1 \
    --c_fa 1 \
    ${trials_dir}/vox1_O_cleaned.kaldi ${trials_dir}/vox1_E_cleaned.kaldi ${trials_dir}/vox1_H_cleaned.kaldi \
    2>&1 | tee /raid/dgxsa/slyne/wespeaker/runtime/server/x86_gpu/embeddings/scores/vox1_cos_result
```

# Perf

We build our engines for 2.02 seconds long audio only by:
```
trtexec --saveEngine=resnet_b1_b128_s200_fp16.trt  --onnx=resnet/resnet_avg_model.onnx --minShapes=feats:1x200x80 --optShapes=feats:64x200x80 --maxShapes=feats:128x200x80 --fp16

trtexec --saveEngine=ecapa_b1_b128_s200_fp16.trt  --onnx=ecapa/ecapa_avg_model.onnx --minShapes=feats:1x200x80 --optShapes=feats:64x200x80 --maxShapes=feats:128x200x80 --fp16
```

* GPU: T4
* resnet: resnet34.

|Engine                              |Throughput (bz=64)| utter/s|
|------------------------------------|------------------|--------|
|resnet_b1_b128_s200_fp16.trt        |39.7842           |2546    |
|ecapa_b1_b128_s200_fp16.trt         |52.958            |3389    |

### Pipeline Perf

In client docker, we may test the whole pipeline performance.
```
cd client/
# generate test input
python3 generate_input.py --audio_file=test.wav --seconds=2.02

perf_analyzer -m speaker -b 1 --concurrency-range 200:1000:200 --input-data=input.json -u localhost:8000
```

|Engine                            | Conccurency | Throughput | Avg Latency(ms) | P99 Latency(ms) |
|----------------------------------|-------------|------------|-----------------|-----------------|
|resnet_b1_b128_s200_fp16.trt      | 200         | 2033       | 98              | 111             |
|                                  | 400         | 2010       | 198             | 208             |
|ecapa_b1_b128_s200_fp16.trt       | 200         | 2647       | 75              | 111             |
|                                  | 400         | 2726       | 147             | 172             |