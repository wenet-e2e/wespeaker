# Wespeaker Python Binding

This is a python binding of Wespeaker.

Wespeaker is a production first and production ready end-to-end speaker verification toolkit.


1. Two models are available: [voxceleb model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx), [cnceleb_model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx)
2. Extract embedding from wav file or wav.scp.
3. Support using `kaldiio` to save embedding.

## Install

Python 3.6+ is required.

``` sh
pip3 install wespeakerruntime
```

## Usage

### extract embedding from wav file

``` python
import sys
import wespeakerruntime as wespeaker
wav_file = sys.argv[1]
inference = wespeaker.Inference(lang='chs')
ans = inference.extract_embedding_wav(wav_file)
print(ans)
```

### extract embedding from wav.scp

``` python
import sys
import wespeakerruntime as wespeaker
wav_scp = sys.argv[1]
inference = wespeaker.Inference(lang='chs')
inference.extract_embedding(wav_scp, 'embed.ark')
```

## Build on Your Local Machine

``` sh
git clone git@github.com:wenet-e2e/wespeaker.git
cd wespeaker/runtime/binding/python
python setup.py install
```