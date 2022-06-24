# Copyright (c) 2022 Xu Xiang
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

import os
import argparse
import pickle
from collections import OrderedDict
# import concurrent.futures

import numpy as np
import scipy.linalg, scipy.cluster

import onnxruntime as ort

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--onnx-model', required=True, help='onnx model')
    parser.add_argument('--wav-scp', required=True, help='wav scp')
    parser.add_argument('--segments', required=True, help='segments')
    args = parser.parse_args()

    return args


def read_segments(segments):
    utt_to_segments = OrderedDict()
    for line in open(segments, 'r'):
        seg, utt, begin, end = line.strip().split()
        begin, end = float(begin), float(end)
        if utt not in utt_to_segments:
            utt_to_segments[utt] = [(seg, begin, end)]
        else:
            utt_to_segments[utt].append((seg, begin, end))

    return utt_to_segments


def compute_fbank(wav_data, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.00001, sample_frequency=16000):
    feats = kaldi.fbank(wav_data,
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=dither,
                        energy_floor=0.0,
                        sample_frequency=sample_frequency,
                        window_type='hamming',
                        htk_compat=True,
                        use_energy=False)
    return feats


def compute_feats(wav_scp, utt_to_segments):
    SAMPLING_RATE = 16000

    # Read WAVE data for each utterance
    for line in open(wav_scp, 'r'):
        utt, wav_path = line.strip().split()
        wav_data, sample_rate = torchaudio.load(wav_path, normalize=False)
        wav_data = wav_data.float()
        assert sample_rate == SAMPLING_RATE
        assert len(wav_data.size()) == 2

        # Compute FBANK features for each segment
        segment_to_feats = OrderedDict()
        segments = utt_to_segments[utt]
        for (seg, begin, end) in segments:

            # Get WAV segment data
            begin_fs = int(begin * sample_rate)
            end_fs = int(end * sample_rate)
            wav_seg_data = wav_data[:, begin_fs:end_fs + 1]

            # Compute FBANK feats
            feats = compute_fbank(wav_seg_data)

            # Apply mean normalization
            feats = feats - torch.mean(feats, dim=0)
            segment_to_feats[seg] = feats.cpu().numpy()

    return segment_to_feats


def compute_embeddings(onnx_model, segment_to_feats, window_frames=150, period_frames=75):

    # Initialize ONNX session
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_model, sess_options=opts, providers=["CPUExecutionProvider"])

    # Extract subsegment-level embeddings
    subsegment_to_embeddings = OrderedDict()
    for segment, feats in segment_to_feats.items():
        feats_frames = feats.shape[0]

        # Extract whole length embeddings for short segments
        if feats_frames <= window_frames:
            subseg_feats = feats[:, :]
            embedding = session.run(
                output_names=['embed_b'],
                input_feed={'x': subseg_feats[None, :]})[0].squeeze()
            subseg = segment + "-{:08d}-{:08d}".format(0, feats_frames)
            subsegment_to_embeddings[subseg] = embedding

        # Extract sliding-window embeddings for long segments
        else:
            for subsegment_begin in range(0, feats_frames - window_frames + period_frames,
                                          period_frames):
                subsegment_end = subsegment_begin + window_frames
                if subsegment_end >= feats_frames:
                    subsegment_end = feats_frames + 2

                # Slice the feature matrix to get subsegment feature
                subseg_feats = feats[subsegment_begin:subsegment_end + 1, :]
                embedding = session.run(
                    output_names=['embed_b'],
                    input_feed={'x': subseg_feats[None, :]})[0].squeeze()
                subseg = segment + "-{:08d}-{:08d}".format(subsegment_begin, subsegment_end)
                subsegment_to_embeddings[subseg] = embedding

    return subsegment_to_embeddings


def groupby_utt(subsegment_to_embeddings):
    utt_to_embeddings = OrderedDict()
    utt_to_subsegments = OrderedDict()
    for subsegment, embedding in subsegment_to_embeddings.items():
        utt = subsegment[:-36]
        if utt not in utt_to_embeddings:
            utt_to_embeddings[utt] = [embedding]
            utt_to_subsegments[utt] = [subsegment]
        else:
            utt_to_embeddings[utt].append(embedding)
            utt_to_subsegments[utt].append(subsegment)
    return list(utt_to_subsegments.values()), list(utt_to_embeddings.values())


def cluster(embeddings, p=0.05, num_spks=None, min_num_spks=1, max_num_spks=10):

    def cosine_similarity(M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return np.dot(M, M.T)

    def prune(M, p):
        m = M.shape[0]
        n = int((1.0 - p) * m)
        for i in range(m):
            indexes = np.argsort(M[i, :])
            low_indexes, high_indexes = indexes[0:n], indexes[n:m]
            M[i, low_indexes] = 0.0
            M[i, high_indexes] = 1.0
        return 0.5 *(M + M.T)

    def laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0.0
        D = np.diag(np.sum(np.abs(M), axis=1))
        return D - M

    def spectral(M, num_spks, min_num_spks, max_num_spks):
        eig_values, eig_vectors = scipy.linalg.eigh(M)
        num_spks = num_spks if num_spks is not None else np.argmax(np.diff(eig_values[:max_num_spks])) + 1
        num_spks = max(num_spks, min_num_spks)
        return eig_vectors[:, :num_spks]

    def kmeans(data):
        k = data.shape[1]
        centroids, labels = scipy.cluster.vq.kmeans2(data, k, minit='++')
        return labels

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(np.array(embeddings))
    # Prune matrix with p interval
    pruned_similarity_matrix = prune(similarity_matrix, p)
    # Compute Laplacian
    laplacian_matrix = laplacian(pruned_similarity_matrix)
    # Compute spectral embeddings
    spectral_embeddings = spectral(laplacian_matrix, num_spks, min_num_spks, max_num_spks)
    # Assign class labels
    labels = kmeans(spectral_embeddings)

    return labels

def main():
    args = get_args()

    # Capable of processing multiple utterances, not just one utterance
    utt_to_segments = read_segments(args.segments)
    segment_to_feats = compute_feats(args.wav_scp, utt_to_segments)
    subsegment_to_embeddings = compute_embeddings(args.onnx_model, segment_to_feats)
    subsegments_list, embeddings_list = groupby_utt(subsegment_to_embeddings)
    for subsegments, labels in zip(subsegments_list, [cluster(e) for e in embeddings_list]):
        [print(s,l) for (s, l) in zip(subsegments, labels)]

if __name__ == '__main__':
    main()
