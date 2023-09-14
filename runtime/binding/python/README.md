# WeSpeaker Python Binding

This is a python binding of WeSpeaker.

WeSpeaker is a production first and production ready end-to-end speaker recognition toolkit.

1. Two onnx models are available: [voxceleb model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx), [cnceleb_model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx)
2. Extract embedding from wav file or feature(Fbank/MFCC).
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
ans = speaker.extract_embedding(wav_file)
print(ans)
```

You can also specify the following parameter in `wespeaker.Speaker`

- `onnx_path` (str, optional): is the path of `onnx model`.
  - Default: onnx model will be downloaded from the server.
- `lang` (str): `chs` for [cnceleb_model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx). `en` for [voxceleb model](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx).
- `inter_op_num_threads` and `intra_op_num_threads` (int): the number of threads during the model runing. For details, please see: https://onnxruntime.ai/docs/

The parameters of `extract_embedding`
- `wav_path` (str): the path of wav
- `resample_rate` (int): resampling rate. Default: 16000
- `num_mel_bins` (int): dimension of fbank. Default: 80
- `frame_length` (int): frame length. Default: 25
- `frame_shift` (int): frame shift. Default: 10
- `cmn` (bool): if true, cepstrum average normalization is applied. Default: True

### Compute cosine score

```python
import wespeakerruntime as wespeaker
speaker = wespeaker.Speaker(lang='chs')
emb1 = speaker.extract_embedding(wav1_path)[0]
emb2 = speaker.extract_embedding(wav2_path)[0]
score = speaker.compute_cosine_score(emb1, emb2)
```
The parameters of `compute_cosine_score`:
- `emb1`(numpy.ndarray): embedding of speaker-1
- `emb2`(numpy.ndarray): embedding of speaker-2

### [Optional] Extract embedding from feature(fbank/MFCC)

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
- `cmn`(bool): if true, cepstrum average normalization is applied. Default: True

### [Optional] Extract embedding from wav.scp

``` python
import sys
import wespeakerruntime as wespeaker
wav_scp = sys.argv[1]
speaker = wespeaker.Speaker(lang='chs')
speaker.extract_embedding_kaldiio(wav_scp, 'embed.ark')
```

The parameters of `extract_embedding_kaldiio`:
- `wav_path` (str): the path of wav
- `embed_ark` (str): the path of `$ouput`.ark
- `resample_rate` (int): resampling rate. Default: 16000
- `num_mel_bins`(int): dimension of fbank. Default: 80
- `frame_length`(int): frame length. Default: 25
- `frame_shift`(int): frame shift. Default: 10
- `cmn`(bool): if true, cepstrum average normalization is applied. Default: True

## Build on Your Local Machine

``` sh
git clone git@github.com:wenet-e2e/wespeaker.git
cd wespeaker/runtime/binding/python
python setup.py install
```
