# Copyright (c) 2022 Horizon Robtics. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ['txt', 'speaker']:
                        example[postfix] = file_obj.read().decode(
                            'utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def shuffle(data, shuffle_size=1500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav, speaker, sample_rate}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav, speaker, sample_rate}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def speaker_to_id(data, spk2id):
    """ Parse speaker id

        Args:
            data: Iterable[{key, wav, speaker, sample_rate}]
            spk2id: Dict[str, int]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'speaker' in sample
        sample['label'] = spk2id[sample['speaker']]
        yield sample


def speed_perturb(data, num_speakers):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed_idx = random.randint(0, 2)
        if speed_idx > 0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speeds[speed_idx])], ['rate',
                                                     str(sample_rate)]])
            sample['wav'] = wav
            sample['label'] = sample['label'] + num_speakers * speed_idx

        yield sample


def get_random_chunk(data, chunk_len):
    # chunking: randomly select a range of size min(chunk_len, len).
    data_len = len(data)
    data_shape = data.shape
    adjust_chunk_len = min(data_len, chunk_len)
    chunk_start = random.randint(0, data_len - adjust_chunk_len)

    data = data[chunk_start:chunk_start + adjust_chunk_len]
    # padding if needed
    if adjust_chunk_len < chunk_len:
        chunk_shape = chunk_len if len(data_shape) == 1 else (chunk_len,
                                                              data.shape[1])
        data = np.resize(data, chunk_shape)  # repeating

    return data


def random_chunk(data, chunk_len=2.0):
    """ the data

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            chunk_len: required chunk size for speaker training, in seconds

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    # Note(Binbin Zhang): We assume the sample rate is 16000,
    #                     frame shift 10ms, frame length 25ms
    chunk_size = (int(chunk_len * 100) - 1) * 160 + 400
    for sample in data:
        assert 'key' in sample
        assert 'wav' in sample
        data_size = sample['wav'].size(1)
        if data_size >= chunk_size:
            chunk_start = random.randint(0, data_size - chunk_size)
            wav = sample['wav'][:, chunk_start:chunk_start + chunk_size]
        else:
            # TODO(Binbin Zhang): Change to pytorch tensor operation
            wav = sample['wav'].numpy()
            new_shape = [wav.shape[0], chunk_size]
            # Resize will repeat copy
            wav = np.resize(wav, new_shape)
            wav = torch.from_numpy(wav)
        sample['wav'] = wav
        yield sample


def add_reverb(data, reverb_source, aug_prob):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            reverb_source: LMDB data source

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            audio = sample['wav'].numpy()[0]
            audio_len = audio.shape[0]
            _, rir_data = reverb_source.random_one()
            rir_io = io.BytesIO(rir_data)
            _, rir_audio = wavfile.read(rir_io)
            rir_audio = rir_audio.astype(np.float32)
            rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
            out_audio = signal.convolve(audio, rir_audio, mode='full')[:audio_len]
            out_audio = torch.from_numpy(out_audio)
            out_audio = torch.unsqueeze(out_audio, 0)
            sample['wav'] = out_audio
        yield sample


def add_noise(data, noise_source, aug_prob):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            noise_source: LMDB data source

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            audio = sample['wav'].numpy()[0]
            audio_len = audio.shape[0]
            audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

            key, noise_data = noise_source.random_one()
            if key.startswith('noise'):
                snr_range = [0, 15]
            elif key.startswith('speech'):
                snr_range = [3, 30]
            elif key.startswith('music'):
                snr_range = [5, 15]
            else:
                snr_range = [0, 15]
            _, noise_audio = wavfile.read(io.BytesIO(noise_data))
            noise_audio = noise_audio.astype(np.float32)
            if noise_audio.shape[0] > audio_len:
                start = random.randint(0, noise_audio.shape[0] - audio_len)
                noise_audio = noise_audio[start:start + audio_len]
            else:
                # Resize will repeat copy
                noise_audio = np.resize(noise_audio, (audio_len, ))
            noise_snr = random.uniform(snr_range[0], snr_range[1])
            noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
            noise_audio = np.sqrt(10**(
                (audio_db - noise_db - noise_snr) / 10)) * noise_audio
            out_audio = audio + noise_audio
            out_audio = torch.from_numpy(out_audio)
            out_audio = torch.unsqueeze(out_audio, 0)
            sample['wav'] = out_audio
        yield sample


def add_noise_reverb(data, noise_source, reverb_source, aug_prob):
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            augtype = random.randint(1, 2)
            if augtype == 1:
                # add reverb
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]
                _, rir_data = reverb_source.random_one()
                rir_io = io.BytesIO(rir_data)
                _, rir_audio = wavfile.read(rir_io)
                rir_audio = rir_audio.astype(np.float32)
                rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
                out_audio = signal.convolve(audio, rir_audio, mode='full')[:audio_len]
                out_audio = torch.from_numpy(out_audio)
                out_audio = torch.unsqueeze(out_audio, 0)
                sample['wav'] = out_audio
            elif augtype == 2:
                # 
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]
                audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

                key, noise_data = noise_source.random_one()
                if key.startswith('noise'):
                    snr_range = [0, 15]
                elif key.startswith('speech'):
                    snr_range = [3, 30]
                elif key.startswith('music'):
                    snr_range = [5, 15]
                else:
                    snr_range = [0, 15]
                _, noise_audio = wavfile.read(io.BytesIO(noise_data))
                noise_audio = noise_audio.astype(np.float32)
                if noise_audio.shape[0] > audio_len:
                    start = random.randint(0, noise_audio.shape[0] - audio_len)
                    noise_audio = noise_audio[start:start + audio_len]
                else:
                    # Resize will repeat copy
                    noise_audio = np.resize(noise_audio, (audio_len, ))
                noise_snr = random.uniform(snr_range[0], snr_range[1])
                noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
                noise_audio = np.sqrt(10**(
                    (audio_db - noise_db - noise_snr) / 10)) * noise_audio
                out_audio = audio + noise_audio
                out_audio = torch.from_numpy(out_audio)
                out_audio = torch.unsqueeze(out_audio, 0)
                sample['wav'] = out_audio

        yield sample


def compute_fbank(data,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          htk_compat=True,
                          use_energy=False)
        # CMN
        mat = mat - torch.mean(mat, dim=0)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.5):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        if random.random() < prob:
            assert 'feat' in sample
            x = sample['feat']
            assert isinstance(x, torch.Tensor)
            y = x.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0
            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
            sample['feat'] = y
        yield sample
