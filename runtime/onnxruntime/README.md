# ONNX backend on WeSpeaker

* Step 1. Export your experiment model to ONNX by https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/export_onnx.py

``` sh
exp=exp  # Change it to your experiment dir
onnx_dir=onnx
python wespeaker/bin/export_onnx.py \
  --config $exp/config.yaml \
  --checkpoint $exp/avg_model.pt \
  --output_model $onnx_dir/final.onnx

# When it finishes, you can find `final.onnx`.
```

* Step 2. Build. The build requires cmake 3.14 or above, and gcc/g++ 5.4 or above.

``` sh
mkdir build && cd build
# 1. no gpu
cmake -DONNX=ON ..
# 2. gpu
# cmake -DONNX=ON -DGPU=ON ..
cmake --build .
```

* Step 3. Testing.

> NOTE: If using GPU, you need to specify the cuda path.
> ```bash
> export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
> export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
> ```

1. the RTF(real time factor) is shown in the console, and embedding will be written to the txt file.
``` sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_scp=your_test_wav_scp
onnx_dir=your_model_dir
embed_out=your_embedding_txt
./build/bin/extract_emb_main \
  --wav_scp $wav_scp \
  --result $embed_out \
  --speaker_model_path $onnx_dir/final.onnx \
  --embedding_size 256 \
  --samples_per_chunk  80000  # 5s

```

> NOTE: samples_per_chunk: samples of one chunk. samples_per_chunk = sample_rate * duration
>
> If samples_per_chunk = -1, compute the embedding of whole sentence;
> else compute embedding with chunk by chunk, and then average embeddings of chunk.

2. Calculate the similarity of two speech.
```sh
export GLOG_logtostderr=1
export GLOG_v=2
onnx_dir=your_model_dir
./build/bin/asv_main \
    --enroll_wav wav1_path \
    --test_wav wav2_path \
    --threshold 0.5 \
    --speaker_model_path $onnx_dir/final.onnx \
    --embedding_size 256
```

## Benchmark
1. RTF
> num_threads = 1
>
> samples_per_chunk = 80000
>
> CPU: Intel(R) Xeon(R) Platinum 8160 CPU @ 2.10GHz

| Model               | Params  | FLOPs    | RTF      |
| :------------------ | :------ | :------- | :------- |
| ECAPA-TDNN (C=512)  | 6.19 M  | 1.04 G   | 0.018351 |
| ECAPA-TDNN (C=1024) | 14.65 M | 2.65 G   | 0.041724 |
| RepVGG-TINY-A0      | 6.26 M  | 4.65 G   | 0.055117 |
| ResNet-34           | 6.63 M  | 4.55 G   | 0.060735 |
| ResNet-50           | 11.13 M | 5.17 G   | 0.073231 |
| ResNet-101          | 15.89 M | 9.96 G   | 0.124613 |
| ResNet-152          | 19.81 M | 14.76 G  | 0.179379 |
| ResNet-221          | 23.79 M | 21.29 G  | 0.267511 |
| ResNet-293          | 28.62 M | 28.10 G  | 0.364011 |
| CAM++               | 7.18 M  | 1.15 G   | 0.022978 |

> num_threads = 1
>
> samples_per_chunk = 80000
>
> CPU: Intel(R) Xeon(R) Platinum 8160 CPU @ 2.10GHz
>
> GPU: NVIDIA 3090

| Model               | Params  | FLOPs    | RTF        |
| :------------------ | :------ | :------- | :--------- |
| ResNet-34           | 6.63 M  | 4.55 G   | 0.00857436 |

2. EER (%)
> onnxruntime: samples_per_chunk=-1.
>
> don't use mean normalization for evaluation embeddings.

| Model          | vox-O | vox-E | vox-H |
| :------------- | ----- | ----- | ----- |
| ResNet-34-pt   | 0.814 | 0.933 | 1.679 |
| ResNet-34-onnx | 0.814 | 0.933 | 1.679 |
