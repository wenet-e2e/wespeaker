# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
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
import random

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio.compliance.kaldi as kaldi
from wespeaker.dataset.processor import (
    get_random_chunk, )


def spk_to_id(data, spk2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav/feat, spk}]
            spk2id: Dict[str, int]

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        if spk2id and ('spk' in sample) and (sample['spk'] in spk2id):
            label = spk2id[sample['spk']]
        else:
            label = -1
        sample['label'] = label
        yield sample


def random_chunk_for_dino(data,
                          global_chunk_len,
                          global_chunk_num,
                          local_chunk_len,
                          local_chunk_num,
                          data_type='shard/raw/feat'):
    """
    Following the strategy in https://arxiv.org/pdf/2104.14294.pdf,
    and https://arxiv.org/abs/2210.15936, several global and local
    chunks are sampled from each utterance for DINO training.
        Args:
            data: Iterable[{key, wav/feat, label}]
            global_chunk_len: chunk length for global chunk
            global_chunk_num: chunk number for global chunk
            local_chunk_len: chunk length for local chunk
            local_chunk_num: chunk number for local chunk

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            sample['feat'] = {'local_chunks': [], 'global_chunks': []}
            for i in range(local_chunk_num):
                sample['feat']['local_chunks'].append(
                    get_random_chunk(feat, local_chunk_len))
            for i in range(global_chunk_num):
                sample['feat']['global_chunks'].append(
                    get_random_chunk(feat, global_chunk_len))
        else:
            assert 'wav' in sample
            wav = sample['wav'][0]
            sample['wav'] = {'local_chunks': [], 'global_chunks': []}
            for i in range(local_chunk_num):
                sample['wav']['local_chunks'].append(
                    get_random_chunk(wav, local_chunk_len).unsqueeze(0))
            for i in range(global_chunk_num):
                sample['wav']['global_chunks'].append(
                    get_random_chunk(wav, global_chunk_len).unsqueeze(0))
        yield sample


def add_reverb(audio, reverb_source, resample_rate=16000):
    """ Add reverb

        Args:
            audio: numpy.array (audio_len, )
            reverb_source: reverb LMDB data source
            resample_rate: resample rate for reverb/noise data
        Returns:
            numpy.array (audio_len, )

    """
    audio_len = audio.shape[0]

    _, rir_data = reverb_source.random_one()
    rir_sr, rir_audio = wavfile.read(io.BytesIO(rir_data))
    rir_audio = rir_audio.astype(np.float32)
    if rir_sr != resample_rate:
        rir_audio = signal.resample(
            rir_audio, int(len(rir_audio) / rir_sr * resample_rate))
    rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
    out_audio = signal.convolve(audio, rir_audio, mode='full')[:audio_len]

    return out_audio


def add_noise(audio, noise_source, resample_rate=16000):
    """ Add reverb

        Args:
            audio: numpy.array (audio_len, )
            noise_source: noise LMDB data source
            resample_rate: resample rate for reverb/noise data
        Returns:
            numpy.array (audio_len, )

    """
    audio_len = audio.shape[0]
    audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

    key, noise_data = noise_source.random_one()
    if key.startswith('noise'):
        snr_range = [0, 15]
    elif key.startswith('speech'):
        snr_range = [10, 30]
    elif key.startswith('music'):
        snr_range = [5, 15]
    else:
        snr_range = [0, 15]
    noise_sr, noise_audio = wavfile.read(io.BytesIO(noise_data))
    noise_audio = noise_audio.astype(np.float32) / (1 << 15)
    if noise_sr != resample_rate:
        # Since the noise audio could be very long, it must be
        # chunked first before resampled (to save time)
        noise_audio = get_random_chunk(
            noise_audio, int(audio_len / resample_rate * noise_sr))
        noise_audio = signal.resample(noise_audio, audio_len)
    else:
        noise_audio = get_random_chunk(noise_audio, audio_len)
    noise_snr = random.uniform(snr_range[0], snr_range[1])
    noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
    noise_audio = np.sqrt(10**(
        (audio_db - noise_db - noise_snr) / 10)) * noise_audio
    out_audio = audio + noise_audio

    return out_audio


def add_reverb_noise(data,
                     reverb_source,
                     noise_source,
                     resample_rate=16000,
                     aug_prob=0.6):
    """ Add reverb & noise aug

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            reverb_source: reverb LMDB data source
            noise_source: noise LMDB data source
            resample_rate: resample rate for reverb/noise data
            aug_prob: aug probability

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """

    def aug_for_an_audio(audio):
        """ Add reverb & noise aug for and audio

            Args:
                audio: torch.Tensor (1, audio_len)
            Returns:
                torch.Tensor (1, audio_len)
        """
        if aug_prob > random.random():
            audio = audio.numpy()[0]
            aug_type = random.randint(1, 2)
            if aug_type == 1:
                # add reverberation
                out_audio = add_reverb(audio, reverb_source, resample_rate)
            else:
                # add additive noise
                out_audio = add_noise(audio, noise_source, resample_rate)

            # normalize into [-1, 1]
            out_audio = out_audio / (np.max(np.abs(out_audio)) + 1e-4)
            return torch.from_numpy(out_audio).unsqueeze(0)
        else:
            return audio

    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if isinstance(sample['wav'], dict):
            # for self supervised training, many chunks are sampled
            # from each utterance.
            # sample['wav'] = {'chunk_type':[chunk1, chunk2, ...], ...}
            for key in sample['wav']:
                for i, audio in enumerate(sample['wav'][key]):
                    sample['wav'][key][i] = aug_for_an_audio(audio)
        else:
            sample['wav'] = aug_for_an_audio(sample['wav'])
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
            Iterable[{key, feat, label, sample_rate}]
    """

    def compute_fbank_for_an_audio(waveform):

        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          use_energy=False)
        return mat

    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        if isinstance(sample['wav'], dict):
            # for self supervised training, many chunks are sampled
            # from each utterance.
            # sample['wav'] = {'chunk_type':[chunk1, chunk2, ...], ...}
            feat_dict = {}
            for key in sample['wav']:
                feat_dict[key] = []
                for waveform in sample['wav'][key]:
                    feat_dict[key].append(compute_fbank_for_an_audio(waveform))
            mat = feat_dict
        else:
            waveform = sample['wav']
            mat = compute_fbank_for_an_audio(waveform)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """ Apply CMVN

        Args:
            data: Iterable[{key, feat, label}]

        Returns:
            Iterable[{key, feat, label}]
    """

    def apply_cmvn_for_a_feat(mat):
        if norm_mean:
            mat = mat - torch.mean(mat, dim=0)
        if norm_var:
            mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
        return mat

    for sample in data:
        assert 'key' in sample
        assert 'feat' in sample
        assert 'label' in sample
        if isinstance(sample['feat'], dict):
            # for self supervised training, many chunks are sampled
            # from each utterance.
            # sample['feat'] = {'chunk_type':[chunk1, chunk2, ...], ...}
            for key in sample['feat']:
                for i, mat in enumerate(sample['feat'][key]):
                    sample['feat'][key][i] = apply_cmvn_for_a_feat(mat)
            mat = sample['feat']
        else:
            mat = sample['feat']
            mat = apply_cmvn_for_a_feat(mat)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            prob: prob of spec_aug

        Returns
            Iterable[{key, feat, label}]
    """

    def spec_aug_for_a_feat(x):
        if random.random() < prob:
            assert isinstance(x, torch.Tensor)
            # y = x.clone().detach()
            y = x.detach()  # inplace operation
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
            return y
        else:
            return x

    for sample in data:
        assert 'feat' in sample
        if isinstance(sample['feat'], dict):
            # for self supervised training, many chunks are sampled
            # from each utterance.
            # sample['feat'] = {'chunk_type':[chunk1, chunk2, ...], ...}
            for key in sample['feat']:
                for i, x in enumerate(sample['feat'][key]):
                    sample['feat'][key][i] = spec_aug_for_a_feat(x)
        else:
            x = sample['feat']
            sample['feat'] = spec_aug_for_a_feat(x)
        yield sample
