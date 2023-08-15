# Copyright (c) 2022, Shuai Wang (wsstriving@gmail.com)
#               2022, Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

from typing import Optional
import numpy as np
import torchaudio
import onnxruntime as ort
import torchaudio.compliance.kaldi as kaldi
from .hub import Hub
import kaldiio


class Speaker:

    def __init__(self,
                 onnx_path: Optional[str] = None,
                 lang: str = 'chs',
                 inter_op_num_threads: int = 1,
                 intra_op_num_threads: int = 1):
        """ Init WeSpeaker Inference based onnxruntime
        Args:
            model_dir: the onnx model
            lang: language type of the model
            inter_op_num_threads and intra_op_num_threads:
                the number of threads during the model running
                For details, please see: https://onnxruntime.ai/docs/
        """
        if onnx_path is None:
            onnx_path = Hub.get_model_by_lang(lang)

        # init onnx model
        so = ort.SessionOptions()
        so.inter_op_num_threads = inter_op_num_threads
        so.intra_op_num_threads = intra_op_num_threads
        self.session = ort.InferenceSession(onnx_path, sess_options=so)

    def _compute_fbank(self,
                       wav_path: str,
                       resample_rate: int = 16000,
                       num_mel_bins: int = 80,
                       frame_length: int = 25,
                       frame_shift: int = 10,
                       dither: float = 0.0,
                       cmn: bool = True):
        """ Extract fbank, simlilar to the one in wespeaker.dataset.processor,
            While integrating the wave reading and CMN.
        """
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != resample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        waveform = waveform * (1 << 15)
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          sample_frequency=resample_rate,
                          window_type='hamming',
                          use_energy=False)
        mat = mat.numpy()
        if cmn:
            # CMN, without CVN
            mat = mat - np.mean(mat, axis=0)
        return mat

    def extract_embedding(self, wav_path: str,
                          resample_rate: int = 16000,
                          num_mel_bins: int = 80,
                          frame_length: int = 25,
                          frame_shift: int = 10,
                          cmn: bool = True):
        """ Extract embedding from wav file, and use fbank features.

        Args:
            wav_path(str): the path of wav
            resample_rate(int): sampling rate
            num_mel_bins(int): dimension of fbank
            frame_length(int): frame length
            frame_shift(int): frame shift
            cmn(bool): if true, cepstrum average normalization is applied
        Return:
            embeddings(numpy.ndarray): [1, emb_dim]
        """
        feats = self._compute_fbank(wav_path,
                                    resample_rate=resample_rate,
                                    num_mel_bins=num_mel_bins,
                                    frame_length=frame_length,
                                    frame_shift=frame_shift,
                                    cmn=cmn)
        feats = np.expand_dims(feats, 0)
        embeddings = self.session.run(output_names=['embs'],
                                      input_feed={'feats': feats})
        return embeddings[0]  # [1, emb_dim]

    def extract_embedding_feat(self, feats, cmn=True):
        """ Extract embedding from feature(fbank).
        Args:
            feats(numpy.ndarray): [B, T, D]
            cmn(bool): if true, cepstrum average normalization is applied
        Returns:
            embeddings(numpy.ndarray): [B, emb_dim]
        """
        assert isinstance(
            feats, np.ndarray), 'NOTE: the type of feats need be np.ndarray.'
        assert len(feats.shape) == 3, "NOTE: the shape of feats is [B, T, D]."
        if cmn:
            # CMN, without CVN
            feats = feats - np.mean(feats, axis=1)
        embeddings = self.session.run(output_names=['embs'],
                                      input_feed={'feats': feats})
        return embeddings[0]  # [B, embed]

    def extract_embedding_kaldiio(self,
                                  wav_scp: str,
                                  embed_ark: str,
                                  resample_rate: int = 16000,
                                  num_mel_bins: int = 80,
                                  frame_length: int = 25,
                                  frame_shift: int = 10,
                                  cmn: bool = True):
        """ Extract embedding from wav.scp, and use fbank features.
        Args:
            wav.scp(str): [utt, wav_path]
            embed_ark(str): output path (kaldi format)
            resample_rate(int): sampling rate
            num_mel_bins(int): dimension of fbank
            frame_length(int): frame length
            frame_shift(int): frame shift
            cmn(bool): if true, cepstrum average normalization is applied
        """
        assert embed_ark[-3:] == "ark"
        embed_scp = embed_ark[:-3] + "scp"
        wav_scp_fin = open(wav_scp, 'r', encoding='utf-8')

        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            for line in wav_scp_fin.readlines():
                line = line.strip().split()
                utt = line[0]
                embed = self.extract_embedding(line[1], resample_rate,
                                               num_mel_bins, frame_length,
                                               frame_shift, cmn)
                writer(utt, embed)
        wav_scp_fin.close()

    def compute_cosine_score(self, emb1, emb2):
        """ Compute cosine score between emb1 and emb2.
        Args:
            emb1(numpy.ndarray): embedding of speaker-1 [B, emb]
            emb2(numpy.ndarray): embedding of speaker-2 [B, emb]
        Return:
            score(float): cosine score [B]
        """
        assert isinstance(emb1, np.ndarray) and isinstance(
            emb2, np.ndarray
        ), "NOTE: the type of emb1 and emb2 need be numpy.ndarray"
        assert len(emb1.shape) == len(
            emb2.shape
        ), "NOTE: the embedding size of emb1 and emb2 need to be equal"
        xx = np.sum(emb1 ** 2, axis=1) ** 0.5
        x = x / xx[:, np.newaxis]
        yy = np.sum(emb2 ** 2, axis=1) ** 0.5
        y = y / yy[:, np.newaxis]
        return np.diagonal(np.dot(x, y.transpose()))
