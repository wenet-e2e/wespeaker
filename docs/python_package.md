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
$ wespeaker --task similarity --audio_file audio.wav --audio_file2 audio2.wav
$ wespeaker --task diarization --audio_file audio.wav  # TODO
```

You can specify the following parameters. (use `-h` for details)

* `-t` or `--task`: embedding/similarity/diarization are supported
    - embedding: extract embedding for an audio and save it into an output file
    - similarity: compute similarity of two audios (in the range of [0, 1])
    - diarization: apply speaker diarization for an input audio (**TODO**)
* `-l` or `--language`: use Chinese/English speaker models
* `--audio_file`: input audio file path
* `--audio_file2`: input audio file2 path, spicifically for the similarity task
* `--resample_rate`: resample rate (default: 16000)
* `--vad`: apply vad or not for the input audios (default: true)
* `--output_file`: output file to save speaker embedding

## Python Programming Usage

``` python
import wespeaker

model = wespeaker.load_model('chinese')

# embedding/similarity/diarization
embedding = model.extract_embedding('audio.wav')
similarity = model.compute_similarity('audio1.wav', 'audio2.wav')
diar_result = model.diarize('audio.wav')  # TODO

# register and recognize
model.register('spk1', 'spk1_audio1.wav')
model.register('spk2', 'spk2_audio1.wav')
model.register('spk3', 'spk3_audio1.wav')
result = model.recognize('spk1_audio2.wav')
```

