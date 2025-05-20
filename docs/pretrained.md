# Pretrained Models in Wespeaker

Besides speaker related tasks, speaker embeddings can be utilized for many related tasks which requires speaker
modeling, such as

- voice conversion
- text-to-speech
- speaker adaptive ASR
- target speaker extraction

For users who would like to verify the SV performance or extract speaker embeddings for the above tasks without
troubling about training the speaker embedding learner, we provide two types of pretrained models.

1. **Checkpoint Model**, with suffix **.pt**, the model trained and saved as checkpoint by WeSpeaker python code, you can
   reproduce our published result with it, or you can use it as checkpoint to continue.

2. **Runtime Model**, with suffix **.onnx**, the `runtime model` is exported by `Onnxruntime` on the `checkpoint model`.

## Model License

The pretrained model in WeNet follows the license of it's corresponding dataset.
For example, the pretrained model on VoxCeleb follows ` Creative Commons Attribution 4.0 International License. `, since
it is used as license of the VoxCeleb dataset, see https://mm.kaist.ac.kr/datasets/voxceleb/.

## Onnx Inference Demo

To use the pretrained model in `pytorch` format, please directly refer to the `run.sh` in corresponding recipe.

As for extracting speaker embeddings from the `onnx` model, the following is a toy example.

```bash
# Download the pretrained model in onnx format and save it as onnx_path
# wav_path is the path to your wave file (16k)
python wespeaker/bin/infer_onnx.py --onnx_path $onnx_path --wav_path $wav_path
```

You can easily adapt `infer_onnx.py` to your application, a speaker diarization example can be found
in [the voxconverse recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxconverse).

## Model List

The model with suffix **LM** means that it is further fine-tuned using large-margin fine-tuning, which could perform better on long audios, e.g. >3s.

### modelscope

| Datasets                                      | Languages | Checkpoint (pt)                                                                                                                                                                                                                     | Runtime Model (onnx)                                                                                                                                                                                                                  |
|-----------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ResNet34](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet34.zip) / [ResNet34_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet34_LM.zip)     | [ResNet34](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet34.onnx) / [ResNet34_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet34_LM.onnx)     |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ResNet152_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet152_LM.zip)  | [ResNet152_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet152_LM.onnx)      |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ResNet221_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet221_LM.zip)    | [ResNet221_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet221_LM.onnx)     |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ResNet293_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet293_LM.zip)    | [ResNet293_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet293_LM.onnx)     |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [CAM++](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_CAM%2B%2B.zip) / [CAM++_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_CAM%2B%2B_LM.zip)                 | [CAM++](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_CAM%2B%2B.onnx) / [CAM++_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_CAM%2B%2B_LM.onnx)          |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ECAPA512](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA512.zip) / [ECAPA512_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA512_LM.zip) / [ECAPA512_DINO](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ecapa512_dino.zip)    | [ECAPA512](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA512.onnx) / [ECAPA512_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA512_LM.onnx)     |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ECAPA1024](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA1024.zip) / [ECAPA1024_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA1024_LM.zip) | [ECAPA1024](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA1024.onnx) / [ECAPA1024_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA1024_LM.onnx) |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [Gemini_DFResnet114_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_gemini_dfresnet114_LM.zip)| [Gemini_DFResnet114_LM](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_gemini_dfresnet114_LM.onnx)  |
| [CNCeleb](../examples/cnceleb/v2/README.md)   | CN        | [ResNet34](https://wenet.org.cn/downloads?models=wespeaker&version=cnceleb_resnet34.zip) / [ResNet34_LM](https://wenet.org.cn/downloads?models=wespeaker&version=cnceleb_resnet34_LM.zip)      | [ResNet34](https://wenet.org.cn/downloads?models=wespeaker&version=cnceleb_resnet34.onnx) / [ResNet34_LM](https://wenet.org.cn/downloads?models=wespeaker&version=cnceleb_resnet34_LM.onnx)         |
| [VoxBlink2](../examples/voxceleb/v2/README.md) | Multilingual        | [SimAMResNet34](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34.zip)  | [SimAMResNet34](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34.onnx)      |
| [VoxBlink2 (pretrain) + VoxCeleb2 (finetune)](../examples/voxceleb/v2/README.md) | Multilingual        | [SimAMResNet34](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34_ft.zip)  |[SimAMResNet34](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34_ft.onnx)  |
| [VoxBlink2](../examples/voxceleb/v2/README.md) | Multilingual        | [SimAMResNet100](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet100.zip)  |[SimAMResNet100](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet100.onnx)|
| [VoxBlink2 (pretrain) + VoxCeleb2 (finetune)](../examples/voxceleb/v2/README.md) | Multilingual        | [SimAMResNet100](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet100_ft.zip)  |[SimAMResNet100](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet100_ft.onnx) |
### huggingface

| Datasets                                      | Languages | Checkpoint (pt)                                                                                                                                                                                                                     | Runtime Model (onnx)                                                                                                                                                                                                                  |
|-----------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ResNet34](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/tree/main) / [ResNet34_LM](https://huggingface.co/Wespeaker/wespeaker-resnet34-LM/tree/main)     | [ResNet34](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/main/voxceleb_resnet34.onnx?download=true) / [ResNet34_LM](https://huggingface.co/Wespeaker/wespeaker-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx?download=true)     |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ResNet152_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet152-LM/tree/main)  | [ResNet152_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet152-LM/resolve/main/voxceleb_resnet152_LM.onnx?download=true)      |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ResNet221_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet221-LM/tree/main)    | [ResNet221_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet221-LM/resolve/main/voxceleb_resnet221_LM.onnx?download=true)     |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ResNet293_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM/tree/main)    | [ResNet293_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM/resolve/main/voxceleb_resnet293_LM.onnx?download=true)     |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [CAM++](https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/tree/main) / [CAM++_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM/tree/main)                 | [CAM++](https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/resolve/main/voxceleb_CAM%2B%2B.onnx?download=true) / [CAM++_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM/resolve/main/voxceleb_CAM%2B%2B_LM.onnx?download=true)          |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ECAPA512](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512/tree/main) / [ECAPA512_LM](https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/tree/main)     | [ECAPA512](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512/resolve/main/voxceleb_ECAPA512.onnx?download=true) / [ECAPA512_LM](https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/main/voxceleb_ECAPA512_LM.onnx?download=true)     |
| [VoxCeleb](../examples/voxceleb/v2/README.md) | EN        | [ECAPA1024](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024/tree/main) / [ECAPA1024_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM/tree/main) | [ECAPA1024](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024/resolve/main/voxceleb_ECAPA1024.onnx?download=true) / [ECAPA1024_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM/resolve/main/voxceleb_ECAPA1024_LM.onnx?download=true) |
| [VoxCeleb](../examples/voxceleb/v2/README.md)   | EN    | [Gemini_DFResnet114_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-gemini-DFresnet114-LM/tree/main)| [Gemini_DFResnet114_LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-gemini-DFresnet114-LM/resolve/main/voxceleb_gemini_dfresnet114_LM.onnx?download=true)  |
| [CNCeleb](../examples/cnceleb/v2/README.md)   | CN        | [ResNet34](https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34/tree/main) / [ResNet34_LM](https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM/tree/main)         | [ResNet34](https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34/resolve/main/cnceleb_resnet34.onnx?download=true) / [ResNet34_LM](https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM/resolve/main/cnceleb_resnet34_LM.onnx?download=true)         |
