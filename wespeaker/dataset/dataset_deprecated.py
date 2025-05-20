# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import numpy as np
import torch
from torch.utils.data import Dataset

import kaldiio
from scipy import signal
from scipy.io import wavfile
import torchaudio.compliance.kaldi as kaldi

from wespeaker.utils.file_utils import read_scp
from wespeaker.dataset.dataset_utils_deprecated import (get_random_chunk,
                                                        speed_perturb,
                                                        spec_augmentation)


class FeatList_LableDict_Dataset(Dataset):
    """
    shuffle wav.scp/feats.scp, load all labels into cpu memory
    """

    def __init__(self, data_list, utt2spkid_dict, whole_utt=False, **kwargs):
        super(FeatList_LableDict_Dataset, self).__init__()
        self.data_list = data_list
        self.length = len(data_list)
        self.utt2spkid_dict = utt2spkid_dict
        self.whole_utt = whole_utt  # True means batch_size=1 !!

        # feat config
        self.raw_wav = kwargs.get('raw_wav', True)
        self.feat_dim = kwargs.get('feat_dim', 80)
        self.num_frms = kwargs.get('num_frms', 200)
        # chunk config (sample rate is 16kHZ)
        self.chunk_len = (self.num_frms -
                          1) * 160 + 400 if self.raw_wav else self.num_frms

        # dataset config (for wav augmentation only)
        if self.raw_wav:
            self.speed_perturb = kwargs.get('speed_perturb', False)
            self.aug_prob = kwargs.get('aug_prob', 0.0)
            self.musan_scp = kwargs.get('musan_scp', '')
            self.rirs_scp = kwargs.get('rirs_scp', '')
            if self.aug_prob > 0.0:
                self.augment_wav = Augment_Wav(self.musan_scp, self.rirs_scp)
        self.spec_aug = kwargs.get('spec_aug', False)

        # used for calculate the spk id after speed perturb
        self.spk_num = len(set(utt2spkid_dict.values()))

    def __getitem__(self, idx):
        utt, data_path = self.data_list[idx]
        spkid = self.utt2spkid_dict[utt] if utt in self.utt2spkid_dict else -1

        speed_perturb_idx = 0
        if self.raw_wav:
            # load wav file
            sr, waveform = wavfile.read(data_path)
            # kaldiio.load_mat() is a little slower than wavfile.read(),
            # but supports cloud io (e.g., kaldiio.load_mat(
            # 'ffmpeg -i http://ip/xxx.wav -ac 1 -ar 16000 -f wav - |'))

            # speed perturb
            if self.speed_perturb:
                speed_perturb_idx = random.randint(0, 2)
                waveform = speed_perturb(waveform,
                                         speed_perturb_idx=speed_perturb_idx)
            # chunk/pad
            if not self.whole_utt:
                waveform = get_random_chunk(waveform, self.chunk_len)
            # augment wav
            if self.aug_prob > random.random():
                waveform = self.augment_wav.process(waveform)
            # make fbank feature
            feat_tensor = kaldi.fbank(torch.FloatTensor(waveform).unsqueeze(0),
                                      num_mel_bins=self.feat_dim,
                                      frame_shift=10,
                                      frame_length=25,
                                      dither=1.0,
                                      sample_frequency=16000,
                                      window_type='hamming',
                                      use_energy=False)
            feat = feat_tensor.detach().numpy()
        else:
            # load feat
            feat = kaldiio.load_mat(data_path)
            # chunk/pad
            if not self.whole_utt:
                feat = get_random_chunk(feat, self.chunk_len)

        # cmn, without cvn
        feat = feat - np.mean(feat, axis=0)  # (T,F)

        # spec augmentation
        if self.spec_aug:
            feat = spec_augmentation(feat)

        return utt, feat, spkid + self.spk_num * speed_perturb_idx

    def __len__(self):
        return self.length


class Augment_Wav:

    def __init__(self, musan_scp, rirs_scp):

        self.noise_snr = {
            'noise': [0, 15],
            'speech': [13, 20],
            'music': [5, 15]
        }
        self.num_noise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}

        self.rir_list = read_scp(rirs_scp)

        # {'noise': noise_list, 'speech': speech_list, 'music': music_list}
        self.noise_dict = {}
        with open(musan_scp, 'r') as fp:
            for line in fp.readlines():
                segs = line.strip().split()
                noise_type = segs[0].split('/')[0]

                if noise_type not in self.noise_dict:
                    self.noise_dict[noise_type] = []
                # utt_name wav_path
                self.noise_dict[noise_type].append((segs[0], segs[1]))

    def additive_noise(self, noise_type, audio):
        """
        :param noise_type: 'noise', 'speech', 'music'
        :param audio: numpy array, (audio_len,)
        """
        audio = audio.astype(np.float32)
        audio_len = audio.shape[0]
        audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

        num_noise = self.num_noise[noise_type]
        noise_idx_list = random.sample(
            self.noise_dict[noise_type],
            random.randint(num_noise[0], num_noise[1]))
        noise_list = []
        for _, noise_path in noise_idx_list:
            _, noise_audio = wavfile.read(noise_path)
            noise_audio = get_random_chunk(noise_audio,
                                           audio_len).astype(np.float32)

            noise_snr = random.uniform(self.noise_snr[noise_type][0],
                                       self.noise_snr[noise_type][1])
            noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
            noise_list.append(
                np.sqrt(10**((audio_db - noise_db - noise_snr) / 10)) *
                noise_audio)

        return np.sum(np.stack(noise_list), axis=0) + audio

    def reverberate(self, audio):
        """
        :param audio: numpy array, (audio_len,)
        """
        audio = audio.astype(np.float32)
        audio_len = audio.shape[0]

        _, rir_wav = random.choice(self.rir_list)
        _, rir_audio = wavfile.read(rir_wav)
        rir_audio = rir_audio.astype(np.float32)
        rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))

        return signal.convolve(audio, rir_audio, mode='full')[:audio_len]

    def process(self, audio):
        augtype = random.randint(1, 4)
        # print("augtype", augtype)
        if augtype == 1:
            audio = self.reverberate(audio)
        elif augtype == 2:
            audio = self.additive_noise('music', audio)
        elif augtype == 3:
            audio = self.additive_noise('speech', audio)
        elif augtype == 4:
            audio = self.additive_noise('noise', audio)

        return audio
