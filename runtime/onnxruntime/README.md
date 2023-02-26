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

* Step 2. Build. The build requires cmake 3.14 or above.

``` sh
mkdir build && cd build
cmake -DONNX=ON ..
cmake --build .
```

* Step 3. Testing.
1. the RTF(real time factor) is shown in the console, and embedding will be written to the txt file.
``` sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_scp=your_test_wav_scp
onnx_dir=your_model_dir
embed_out=your_embedding_txt
./build/bin/extract_emb_main \
  --wav_list $wav_scp \
  --result $embed_out \
  --speaker_model_path $onnx_dir/final.onnx
```

2. Calculate the similarity of two speech.
```sh
export GLOG_logtostderr=1
export GLOG_v=2
onnx_dir=your_model_dir
./build/bin/asv_main \
    --enroll_wav wav1_path \
    --test_wav wav2_path \
    --threshold 0.5 \
    --speaker_model_path $onnx_dir/final.onnx
```
