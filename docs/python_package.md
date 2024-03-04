# Python Package


## Install

``` sh
pip install git+https://github.com/wenet-e2e/wespeaker.git
```

for development install:

``` sh
git clone https://github.com/wenet-e2e/wespeaker.git
cd wespeaker
pip install -e .
```

## Command line Usage

``` sh
$ wespeaker --task embedding --audio_file audio.wav --output_file embedding.txt
$ wespeaker --task embedding_kaldi --wav_scp wav.scp --output_file /path/to/embedding
$ wespeaker --task similarity --audio_file audio.wav --audio_file2 audio2.wav
$ wespeaker --task diarization --audio_file audio.wav
```

You can specify the following parameters. (use `-h` for details)

* `-t` or `--task`: five tasks are supported now
    - embedding: extract embedding for an audio and save it into an output file
    - embedding_kaldi: extract embeddings from kaldi-style wav.scp and save it to ark/scp files.
    - similarity: compute similarity of two audios (in the range of [0, 1])
    - diarization: apply speaker diarization for an input audio
    - diarization_list: apply speaker diarization for a kaldi-style wav.scp
* `-l` or `--language`: use Chinese/English speaker models
* `-p` or `--pretrain`: the path of pretrained model, `avg_model.pt` and `config.yaml` should be contained
* `-g` or `--gpu`: use GPU for inference, number $< 0$ means using CPU
* `--campplus`: use [`campplus_cn_common_200k` of damo](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary)
* `--eres2net`: use [`res2net_cn_common_200k` of damo](https://www.modelscope.cn/models/iic/speech_eres2net_sv_zh-cn_16k-common/summary)
* `--audio_file`: input audio file path
* `--audio_file2`: input audio file2 path, specifically for the similarity task
* `--wav_scp`: input wav.scp file in kaldi format (each line: key wav_path)
* `--resample_rate`: resample rate (default: 16000)
* `--vad`: apply vad or not for the input audios (default: true)
* `--output_file`: output file to save speaker embedding, if you use kaldi wav_scp, output will be `output_file.ark` and `output_file.scp`

### Pretrained model support
We provide different pretrained models, which can be found at [pretrained models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)

**Warning** If you want to use the models provided in the above link, be sure to rename the model and config file to `avg_model.pt` and `config.yaml`

By default, specifying the `language` option will download the pretrained models as

* english: `ResNet221_LM` pretrained on VoxCeleb
* chinese: `ResNet34_LM` pretrained on CnCeleb

if you want to use other pretrained models, please use the `-p` or `--pretrain` to specify the directory containing `avg_model.pt` and `config.yaml`,
which can either be the ones we provided and trained by yourself.

## Python Programming Usage

``` python
import wespeaker

model = wespeaker.load_model('chinese')
# set_gpu to enable the cuda inference, number < 0 means using CPU
model.set_gpu(0)

# embedding/embedding_kaldi/similarity/diarization
embedding = model.extract_embedding('audio.wav')
utt_names, embeddings = model.extract_embedding_list('wav.scp')
similarity = model.compute_similarity('audio1.wav', 'audio2.wav')
diar_result = model.diarize('audio.wav')

# register and recognize
model.register('spk1', 'spk1_audio1.wav')
model.register('spk2', 'spk2_audio1.wav')
model.register('spk3', 'spk3_audio1.wav')
result = model.recognize('spk1_audio2.wav')
```

