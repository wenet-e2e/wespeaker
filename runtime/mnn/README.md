# MNN backend on WeSpeaker

* Step 1. Export your experiment model to MNN

First, export your experiment model to ONNX by [export_onnx.py](../../wespeaker/bin/export_onnx.py).

``` sh
# 1. dynamic shape
python wespeaker/bin/export_onnx.py \
  --config config.yaml \
  --checkpoint model.pt \
  --output_model model.onnx
  # When it finishes, you can find `model.onnx`.
# 2. static shape
# python wespeaker/bin/export_onnx.py \
#   --config config.yaml \
#   --checkpoint model.pt \
#   --output_model model.onnx \
#   --num_frames 198
```

Second, export ONNX to MNN by [export_mnn.py](../../wespeaker/bin/export_mnn.py).

``` sh
# 1. dynamic shape
python wespeaker/bin/export_mnn.py \
  --onnx_model model.onnx \
  --output_model model.mnn
# When it finishes, you can find `model.mnn`.
# 2. static shape
# python wespeaker/bin/export_mnn.py \
#   --onnx_model model.onnx \
#   --output_model model.mnn \
#   --num_frames 198
```

* Step 2. Build. The build requires cmake 3.14 or above, and gcc/g++ 5.4 or above.

``` sh
mkdir build && cd build
# 1. normal
cmake  ..
# 2. minimum libs
# cmake -DMINI_LIBS=ON ..
cmake --build .
```

* Step 3. Testing.

1. the RTF(real time factor) is shown in the console, and embedding will be written to the txt file.
``` sh
export GLOG_logtostderr=1
export GLOG_v=2
./build/bin/extract_emb_main \
  --wav_scp wav.scp \
  --result embedding.txt \
  --speaker_model_path model.mnn \
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
./build/bin/asv_main \
    --enroll_wav wav1_path \
    --test_wav wav2_path \
    --threshold 0.5 \
    --speaker_model_path model.mnn \
    --embedding_size 256
```

## Benchmark

1. RTF
> num_threads = 1
>
> samples_per_chunk = 32000
>
> Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz

| Model               | Params  | FLOPs    | engine        | RTF      |
| :------------------ | :------ | :------- | :------------ | :------- |
| ResNet-34           | 6.63 M  | 4.55 G   | onnxruntime   | 0.1377   |
| ResNet-34           | 6.63 M  | 4.55 G   | mnn           | 0.1333   |
| ResNet-34           | 6.63 M  | 4.55 G   | mnn mini_libs | 0.05262  |
