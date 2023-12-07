# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import multiprocessing
from multiprocessing import Pool

import tritonclient.grpc as grpcclient
from tritonclient.utils import *
import numpy as np
import soundfile
import argparse
import os
import kaldiio


class SpeakerClient(object):

    def __init__(self, triton_client, model_name, protocol_client):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name

    def recognize(self, wav_path, client_index):
        # We send batchsize=1 data to server
        # BatchSize > 1 is also ok but you need to take care of
        # padding.
        waveform, sample_rate = soundfile.read(wav_path)
        # trim audio that is longer than 30 seconds
        # there are many audio samples in vox1 longer than 30s
        # we found the EER will not grow a lot when trimming the audio to 30s
        max_length = 30 * 16000  # 30 * 16000
        cur_length = min(max_length, len(waveform))
        if len(waveform) > cur_length:
            print(len(waveform) // 16000)
        input = np.zeros((1, cur_length), dtype=np.float32)
        input[0][0:cur_length] = waveform[0:cur_length]
        inputs = [
            self.protocol_client.InferInput("WAV", input.shape,
                                            np_to_triton_dtype(input.dtype))
        ]
        inputs[0].set_data_from_numpy(input)
        outputs = [grpcclient.InferRequestedOutput("EMBEDDINGS")]
        response = self.triton_client.infer(self.model_name,
                                            inputs,
                                            request_id=str(client_index),
                                            outputs=outputs)
        return response.as_numpy("EMBEDDINGS")[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is '
                        'localhost:8001.')
    parser.add_argument('--model_name',
                        required=False,
                        default='speaker',
                        help='the model to send request to')
    parser.add_argument('--wavscp',
                        type=str,
                        required=False,
                        default=None,
                        help='audio_id \t absolute_wav_path')
    parser.add_argument('--output_directory',
                        type=str,
                        required=False,
                        default=None,
                        help='audio_id \t text')
    parser.add_argument('--data_dir',
                        type=str,
                        required=False,
                        default=None,
                        help='data dir will be append to audio file if given')
    parser.add_argument('--audio_file',
                        type=str,
                        required=False,
                        default=None,
                        help='single wav file')

    FLAGS = parser.parse_args()

    # load data
    audio_wavpath = []
    if FLAGS.audio_file is not None:
        path = FLAGS.audio_file
        if FLAGS.data_dir:
            path = os.path.join(FLAGS.data_dir, path)
        if os.path.exists(path):
            audio_wavpath = [(FLAGS.audio_file, path)]
    elif FLAGS.wavscp is not None:
        with open(FLAGS.wavscp, "r", encoding="utf-8") as f:
            for line in f:
                aid, path = line.strip().split()
                if FLAGS.data_dir:
                    path = os.path.join(FLAGS.data_dir, path)
                audio_wavpath.append((aid, path))

    num_workers = multiprocessing.cpu_count() // 2

    def single_job(li):
        idx, audio_files = li
        dir_name = os.path.dirname(FLAGS.output_directory)  # get the path
        if not os.path.exists(dir_name) and (dir_name != ''):
            os.makedirs(dir_name)

        embed_ark = os.path.abspath(dir_name) + "/xvector_{:0>3d}.ark".format(
            idx)
        embed_scp = embed_ark[:-3] + "scp"

        with grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:
            protocol_client = grpcclient
            speech_client = SpeakerClient(triton_client, FLAGS.model_name,
                                          protocol_client)

            with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                     embed_scp) as writer:
                for li in audio_files:
                    utt, wavpath = li
                    embed = speech_client.recognize(wavpath, idx)
                    writer(utt, embed)

        return predictions

    # start to do inference
    # Group requests in batches
    predictions = []
    tasks = []
    splits = np.array_split(audio_wavpath, num_workers)

    for idx, per_split in enumerate(splits):
        cur_files = per_split.tolist()
        tasks.append((idx, cur_files))

    with Pool(processes=num_workers) as pool:
        predictions = pool.map(single_job, tasks)
