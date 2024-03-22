# Pretrained Models in Wespeaker

Besides speaker related tasks, speaker embeddings can be utilized for many related tasks which requires speaker modeling, such as

- voice conversion
- text-to-speech
- speaker adaptive ASR
- target speaker extraction

For users who would like to verify the SV performance or extract speaker embeddings for the above tasks without troubling about training the speaker embedding learner, we provide two types of pretrained models.

1. **Checkpoint Model**, with suffix **.pt**, the model trained and saved as checkpoint by WeNet python code, you can reproduce our published result with it, or you can use it as checkpoint to continue.

2. **Runtime Model**, with suffix **.onnx**, the `runtime model` is exported by `Onnxruntime` on the `checkpoint model`.



## Model License

The pretrained model in WeNet follows the license of it's corresponding dataset.
For example, the pretrained model on VoxCeleb follows ` Creative Commons Attribution 4.0 International License. `, since it is used as license of the VoxCeleb dataset, see https://mm.kaist.ac.kr/datasets/voxceleb/.

## Onnx Inference Demo
To use the pretrained model in `pytorch` format, please directly refer to the `run.sh` in corresponding recipe.

As for extracting speaker embeddings from the `onnx` model, the following is a toy example.

```bash
# Download the pretrained model in onnx format and save it as onnx_path
# wav_path is the path to your wave file (16k)
python wespeaker/bin/infer_onnx.py --onnx_path $onnx_path --wav_path $wav_path
```

You can easily adapt `infer_onnx.py` to your application, a speaker diarization example can be found in [the voxconverse recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxconverse)

## Model List

| Datasets  | Languages     |  Checkpoint (pt) | Runtime Model (onnx)     |
|---    |---    |---   |---   |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [ResNet34](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.zip) / [ResNet34_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.zip) | [ResNet34](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.onnx) / [ResNet34_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx)  |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [ResNet152_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet152_LM.zip)| [ResNet152_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet152_LM.onnx)  |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [ResNet221_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet221_LM.zip)| [ResNet221_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet221_LM.onnx)  |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [ResNet293_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet293_LM.zip)| [ResNet293_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet293_LM.onnx)  |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [CAM++](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_CAM++.zip) / [CAM++_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_CAM++_LM.zip) | [CAM++](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_CAM++.onnx) / [CAM++_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_CAM++_LM.onnx)  |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [ECAPA512](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.zip) / [ECAPA512_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512_LM.zip) | [ECAPA512](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.onnx) / [ECAPA512_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512_LM.onnx)  |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [ECAPA1024](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA1024.zip) / [ECAPA1024_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA1024_LM.zip) | [ECAPA1024](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA1024.onnx) / [ECAPA1024_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA1024_LM.onnx)  |
| [CNCeleb](../examples/cnceleb/v2/README.md)   | CN    | [ResNet34](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34.zip) / [ResNet34_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.zip)  | [ResNet34](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34.onnx) / [ResNet34_LM](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx) |

