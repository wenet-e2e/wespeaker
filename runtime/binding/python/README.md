# Wespeaker Python Binding

This is a python binding of Wespeaker.

Wespeaker is a production first and production ready end-to-end speaker recognition toolkit.


1. Two onnx models are available: [voxceleb model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx), [cnceleb_model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx)
2. Extract embedding from wav file or wav.scp.
3. Support using `kaldiio` to save embedding.

## Install

Python 3.6+ is required.

``` sh
pip3 install wespeakerruntime
```

## Usage

### Extract embedding from wav file

``` python
import sys
import wespeakerruntime as wespeaker
wav_file = sys.argv[1]
speaker = wespeaker.Speaker(lang='chs')
ans = speaker.extract_embedding_wav(wav_file)
print(ans)
```

You can also specify the following parameter in `wespeaker.Speaker`

- `onnx_path` (str, optional): is the path of `onnx model`.
  - Default: onnx model will be downloaded from the server.
- `lang` (str): `chs` for [cnceleb_model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx). `en` for [voxceleb model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx).
- `inter_op_num_threads` and `intra_op_num_threads` (int): the number of threads during the model runing. For details, please see: https://onnxruntime.ai/docs/

The parameters of `extract_embedding_wav`
- `wav_path` (str): the path of wav
- `resample_rate` (int): resampling rate. Default: 16000

### Extract embedding from wav.scp

``` python
import sys
import wespeakerruntime as wespeaker
wav_scp = sys.argv[1]
speaker = wespeaker.Speaker(lang='chs')
speaker.extract_embedding(wav_scp, 'embed.ark')
```

The parameters of `extract_embedding`:
- `wav_path` (str): the path of wav
- `embed_ark` (str): the path of `$ouput`.ark
- `resample_rate` (int): resampling rate. Default: 16000

### Extract embedding from feature(fbank)

``` python
import sys
import wespeakerruntime as wespeaker
feat = your_fbank
speaker = wespeaker.Speaker(lang='chs')
ans = speaker.extract_embedding_feat(feat)
print(ans)
```

The parameters of `extract_embedding_feat`:
- `feats`(numpy.ndarray): the shape is [B, T, D].

### Compute cosine score

```python
import wespeakerruntime as wespeaker
speaker = wespeaker.Speaker(lang='chs')
emb1 = speaker.extract_embedding_wav(wav1_path)
emb2 = speaker.extract_embedding_wav(wav2_path)
score = speaker.compute_cosine_score(emb1, emb2)
```
The parameters of `compute_cosine_score`:
- `emb1`(numpy.ndarray): embedding of speaker-1
- `emb2`(numpy.ndarray): embedding of speaker-2


## Build on Your Local Machine

``` sh
git clone git@github.com:wenet-e2e/wespeaker.git
cd wespeaker/runtime/binding/python
python setup.py install
```