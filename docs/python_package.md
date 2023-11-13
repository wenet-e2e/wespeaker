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
$ wespeaker --task embedding --audio_file audio.wav --output_file embedding.txt -g 0
$ wespeaker --task embedding_kaldi --wav_scp wav.scp --output_file /path/to/embedding -g 0
$ wespeaker --task similarity --audio_file audio.wav --audio_file2 audio2.wav -g 0
$ wespeaker --task diarization --audio_file audio.wav -g 0  # TODO
```

You can specify the following parameters. (use `-h` for details)

* `-t` or `--task`: embedding/embedding_kaldi/similarity/diarization are supported
    - embedding: extract embedding for an audio and save it into an output file
    - embedding_kaldi: extract embeddings from kaldi-style wav.scp and save it to ark/scp files.
    - similarity: compute similarity of two audios (in the range of [0, 1])
    - diarization: apply speaker diarization for an input audio (**TODO**)
* `-l` or `--language`: use Chinese/English speaker models
* `-g` or `--gpu`: use GPU for inference, number $< 0$ means using CPU
* `--audio_file`: input audio file path
* `--audio_file2`: input audio file2 path, specifically for the similarity task
* `--wav_scp`: input wav.scp file in kaldi format (each line: key wav_path)
* `--resample_rate`: resample rate (default: 16000)
* `--vad`: apply vad or not for the input audios (default: true)
* `--output_file`: output file to save speaker embedding, if you use kaldi wav_scp, output will be `output_file.ark` and `output_file.scp`


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
diar_result = model.diarize('audio.wav')  # TODO

# register and recognize
model.register('spk1', 'spk1_audio1.wav')
model.register('spk2', 'spk2_audio1.wav')
model.register('spk3', 'spk3_audio1.wav')
result = model.recognize('spk1_audio2.wav')
```

