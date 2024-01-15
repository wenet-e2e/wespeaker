# WeSpeaker & Horizon BPU (Cross Compile)

* Step 1. Setup environment (install horizon packages and cross compile tools) in the PC.

```sh
# Conda env (This conda env is only used for converting bpu models, not for training torch models,
#   It's OK to install cpu-version pytorch)
conda create -n horizonbpu python=3.8
conda activate horizonbpu
git clone https://github.com/wenet-e2e/wespeaker.git
cd wespeaker/runtime/horizonbpu
pip install -r ../../requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install torch==1.13.0 torchaudio==0.13.0 torchvision==0.14.0 onnx onnxruntime -i https://mirrors.aliyun.com/pypi/simple

# Horizon packages
wget https://gitee.com/xcsong-thu/toolchain_pkg/releases/download/resource/wheels.tar.gz
tar -xzf wheels.tar.gz
pip install wheels/* -i https://mirrors.aliyun.com/pypi/simple

# Cross compile tools
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

* Step 2. Build extract_emb_main. It requires cmake 3.14 or above. and Send the binary/libraries to Horizon X3PI.

```bash
# Assume current dir is `wespeaker/runtime/horizonbpu`
cmake -B build -DBPU=ON -DONNX=OFF -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.toolchain.cmake
cmake --build build

# Send binary and libraries
export BPUIP=xxx.xxx.xxx.xxx
export DEMO_PATH_ON_BOARD=/path/to/demo
scp -r build/bin/ root@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/dnn/*j3*/*/*/lib/libdnn.so root@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/easy_dnn/*j3*/*/*/lib/libeasy_dnn.so root@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/hlog/*j3*/*/*/lib/libhlog.so root@$BPUIP:$DEMO_PATH_ON_BOARD
```

* Step 3. Export model to ONNX and convert ONNX to Horizon .bin and Send the model/test_wav to Horizon X3PI.


```bash
# stage 1: export onnx
export WESPEAKER_DIR=$PWD/../../
export PYTHONPATH=$WESPEAKER_DIR:$PYTHONPATH
mkdir -p output
python $WESPEAKER_DIR/wespeaker/bin/export_onnx_bpu.py \
  --config the_path_train_yaml \
  --checkpoint the_path_pt_model \
  --output_model the_path_onnx_model \
  --num_frames num_frames
# egs:
# python $WESPEAKER_DIR/wespeaker/bin/export_onnx_bpu.py \
#   --config resnet_34/train.yaml \
#   --checkpoint resnet_34/avg_model.pt \
#   --output_model resnet_34/resnet34.onnx \
#   --num_frames 198

# stage 2: export bin model
python $WESPEAKER_DIR/tools/onnx2horizonbin.py \
  --config the_path_train_yaml \
  --output_dir output_dir \
  --cali_datalist data_list \
  --onnx_path fp_onnx_path \
  --cali_data_type "shard/raw" \
  --input_name input_name \
  --input_shape input_shape

# egs:
# python $WESPEAKER_DIR/tools/onnx2horizonbin.py \
#   --config resnet_34/config.yaml \
#   --output_dir output/ \
#   --cali_datalist cali_data/raw.list \
#   --onnx_path resnet_34/resnet34.onnx \
#   --cali_data_type "raw" \
#   --input_name "feats" \
#   --input_shape "1x1x198x80"

# scp test wav file
scp test_wav.wav root@$BPUIP:$DEMO_PATH_ON_BOARD
# scp bpu models
scp ./output/hb_makertbin_output_speaker/speaker.bin root@$BPUIP:$DEMO_PATH_ON_BOARD
```

* Step 4. Testing on X3PI, the RTF(real time factor) is shown in Horizon X3PI's console.

```bash
cd /path/to/demo
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH \
export GLOG_logtostderr=1
export GLOG_v=2
wav_scp=your_test_wav_scp
embed_out=your_embedding_txt
./build/bin/extract_emb_main \
  --wav_scp $wav_scp \
  --result $embed_out \
  --speaker_model_path speaker.bin \
  --embedding_size 256 \
  --SamplesPerChunk  32000  # 2s
```
