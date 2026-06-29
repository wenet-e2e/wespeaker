# Copyright (c) 2026 Bosen Xu (2332974001@qq.com)
# Based on original code from PalabraAI/redimnet2
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import windows


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(low_freq=20, high_freq=7600, nfilt=80, nfft=512, samplerate=16000):
    lowmel = hz2mel(low_freq)
    highmel = hz2mel(high_freq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    lower_edge_mel = melpoints[:-2].reshape(1, -1)
    center_mel = melpoints[1:-1].reshape(1, -1)
    upper_edge_mel = melpoints[2:].reshape(1, -1)
    spectrogram_bins_mel = hz2mel(np.linspace(0, samplerate // 2, nfft))[1:].reshape(-1, 1)
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.vstack([np.zeros((1, nfilt)), mel_weights_matrix])[:, :].astype('float32')


class NormalizeAudio(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return (x - x.mean(dim=2, keepdims=True)) / (x.std(dim=2, keepdims=True, unbiased=False) + self.eps)


class PreEmphasis(nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10), freq_start_bin=0):
        super().__init__()
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.freq_start_bin = freq_start_bin

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width
        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(self.freq_start_bin, max(1, D - mask_len.max()),
                                 (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class SpectralFeaturesTF(nn.Module):
    def __init__(
            self,
            frame_length=400,
            frame_step=160,
            fft_length=512,
            sample_rate=16000,
            window='hann',
            normalize_spectrogram=False,
            normalize_signal=False,
            eps=1e-8,
            mode='melbanks',
            low_freq=20,
            high_freq=7600,
            num_bins=80,
            log_mels=True,
            fft_mode='abs',
            sqrt_real_imag=False,
            return_img=False,
            **kwargs):
        super().__init__()
        self.length = frame_length
        self.shift = frame_step
        self.sqrt_real_imag = sqrt_real_imag
        self.normalize_spectrogram = normalize_spectrogram
        self.normalize_signal = normalize_signal
        self.window = window
        self.eps = eps
        self.nfft = fft_length if fft_length else frame_length
        self.samplerate = sample_rate
        self.features = mode
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_bins = num_bins
        self.return_img = return_img
        if mode in ['melbanks', 'mfcc']:
            fft_mode = 'abs'
        self.fft_mode = fft_mode
        self.log_mels = log_mels
        self.build()

    def build(self):
        if self.window:
            if self.window == 'hamming':
                self.window = windows.hamming(self.length)
            elif self.window in ['hann', 'hanning']:
                self.window = np.array([0.5 - 0.5 * (np.cos((2 * np.pi * l) / (self.length - 1)))
                                       for l in range(self.length)])
            elif self.window == 'sqrt_hann':
                self.window = np.array([0.5 - 0.5 * (np.cos((2 * np.pi * l) / (self.length - 1)))
                                       for l in range(self.length)]) ** 0.5
            elif self.window == 'kaiser':
                self.window = windows.kaiser(self.length)
            else:
                self.window = np.ones(self.length)
        self.window = self.window.astype("float32")
        real_kernel = np.asarray([np.cos(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft)
                                 for n in range(self.nfft)]).astype("float32").T
        self.real_kernel = real_kernel[:self.length, :self.nfft // 2]
        if self.window is not None:
            self.real_kernel *= self.window[:, None]
        self.real_kernel = self.real_kernel[:, None, :]
        image_kernel = np.asarray([np.sin(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft)
                                  for n in range(self.nfft)]).astype("float32").T
        self.image_kernel = image_kernel[:self.length, :self.nfft // 2]
        if self.window is not None:
            self.image_kernel *= self.window[:, None]
        self.image_kernel = self.image_kernel[:, None, :]
        self.register_buffer('real_kernel_pt', torch.from_numpy(self.real_kernel).permute(2, 1, 0).float())
        self.register_buffer('image_kernel_pt', torch.from_numpy(self.image_kernel).permute(2, 1, 0).float())
        if self.features in ['melbanks']:
            linear_to_mel_weight_matrix = get_filterbanks(
                nfilt=self.num_bins,
                nfft=self.nfft // 2,
                samplerate=self.samplerate,
                low_freq=self.low_freq,
                high_freq=self.high_freq)
            linear_to_mel_weight_matrix = linear_to_mel_weight_matrix[:, :, None]
            self.register_buffer('melbanks_pt', torch.from_numpy(linear_to_mel_weight_matrix).permute(1, 0, 2).float())

    def forward(self, inputs):
        dtype = inputs.dtype
        inputs = inputs.float()
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if self.normalize_signal:
            inputs = (inputs - inputs.mean(dim=2, keepdims=True)) / \
                (inputs.std(dim=2, keepdims=True, unbiased=False) + self.eps)
        real_part = F.conv1d(inputs, self.real_kernel_pt, stride=self.shift, padding=self.shift // 2)
        imag_part = F.conv1d(inputs, self.image_kernel_pt, stride=self.shift, padding=self.shift // 2)
        if self.features == 'complex':
            return [real_part, imag_part]
        fft = torch.square(real_part) + torch.square(imag_part)
        if self.sqrt_real_imag:
            fft = torch.sqrt(fft)
        feat = fft.clip(self.eps, 1 / self.eps)
        if self.fft_mode == 'log':
            feat = torch.log(feat)
        if self.features in ['melbanks']:
            mel_spectrograms = F.conv1d(feat, self.melbanks_pt, stride=1, padding=0)
            mel_spectrograms = mel_spectrograms.clip(self.eps, 1 / self.eps)
            if self.log_mels:
                feat = torch.log(mel_spectrograms)
            else:
                feat = mel_spectrograms
        if self.normalize_spectrogram:
            feat = (feat - feat.mean(dim=(1, 2), keepdims=True)) / \
                (feat.std(dim=(1, 2), keepdims=True, unbiased=False) + self.eps)
        if self.return_img:
            feat = feat[:, None, :, :]
        return feat.to(dtype)


class TFMelBanks(nn.Module):
    def __init__(
            self,
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=80,
            do_spec_aug=False,
            norm_signal=False,
            do_preemph=True,
            freq_start_bin=0,
            freq_mask_width=(
                0,
                8),
            time_mask_width=(0, 10),
            eps=1e-8):
        super(TFMelBanks, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            NormalizeAudio(eps) if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),
            SpectralFeaturesTF(
                frame_length=win_length,
                frame_step=hop_length,
                fft_length=n_fft,
                sample_rate=sample_rate,
                window='hamming',
                normalize_spectrogram=False,
                normalize_signal=False,
                eps=eps,
                mode='melbanks',
                low_freq=f_min,
                high_freq=f_max,
                num_bins=n_mels,
                log_mels=False,
                fft_mode='abs',
                sqrt_real_imag=False,
                return_img=False))
        self.eps = eps
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width)
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x) + self.eps
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    x = self.specaug(x)
        return x.to(xdtype)


class TFSpectrogram(nn.Module):
    def __init__(
            self,
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=80,
            window='hanning',
            normalize_spectrogram=False,
            normalize_signal=False,
            mode='fft',
            fft_mode='abs',
            pool_freqs=(
                2,
                1),
            do_spec_aug=False,
            norm_signal=False,
            do_preemph=True,
            freq_start_bin=0,
            num_apply_spec_aug=1,
            freq_mask_width=(0, 8),
            time_mask_width=(0, 10),
            eps=1e-8):
        super(TFSpectrogram, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.spectrogram = torch.nn.Sequential(
            NormalizeAudio() if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),
            SpectralFeaturesTF(
                frame_length=win_length,
                frame_step=hop_length,
                fft_length=n_fft,
                sample_rate=sample_rate,
                window=window,
                eps=eps,
                mode=mode,
                low_freq=f_min,
                high_freq=f_max,
                num_bins=n_mels,
                normalize_spectrogram=False,
                normalize_signal=False,
                fft_mode='abs',
                log_mels=False,
                sqrt_real_imag=False,
                return_img=False))
        if pool_freqs is not None:
            self.pool_freq = nn.AvgPool2d(pool_freqs, stride=pool_freqs)
        else:
            self.pool_freq = nn.Identity()
        self.eps = eps
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width)
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.spectrogram(x) + self.eps
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    for _ in range(self.num_apply_spec_aug):
                        x = self.specaug(x)
                x = self.pool_freq(x.unsqueeze(1))
        return x.to(xdtype)


class TFMelFrontend(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 win_length=400,
                 hop_length=160,
                 f_min=20,
                 f_max=7600,
                 n_mels=80,
                 do_spec_aug=False,
                 norm_signal=False,
                 do_preemph=True,
                 freq_mask_width=(0, 8),
                 time_mask_width=(0, 10),
                 eps=1e-8,
                 **kwargs):
        super().__init__()
        self.n_mels = n_mels
        self.spec = TFMelBanks(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            do_spec_aug=do_spec_aug,
            norm_signal=norm_signal,
            do_preemph=do_preemph,
            freq_mask_width=freq_mask_width,
            time_mask_width=time_mask_width,
            eps=eps,
        )

    def output_size(self):
        return self.n_mels

    def forward(self, wavs, wavs_len=None):
        features = self.spec(wavs)
        return features, None
