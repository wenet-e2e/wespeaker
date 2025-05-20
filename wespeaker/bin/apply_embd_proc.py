# Copyright (c) 2024 Johan Rohdin (rohdin@fit.vutbr.cz)
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

import argparse
import kaldiio
import numpy as np
from wespeaker.utils.embedding_processing import EmbeddingProcessingChain

if __name__ == '__main__':
    """
    xxx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        type=str,
                        default='',
                        help='Path to processing chain.')
    parser.add_argument('--input',
                        type=str,
                        default='',
                        help='Input scp file.')
    parser.add_argument('--output',
                        type=str,
                        default='',
                        help='Output scp/ark file.')
    args = parser.parse_args()

    processingChain = EmbeddingProcessingChain()
    processingChain.load(args.path)

    embd = []
    utt = []
    for k, v in kaldiio.load_scp_sequential(args.input):
        utt.append(k)
        embd.append(v)
    embd = np.array(embd)
    utt = np.array(utt)

    print("Read {} embeddings of dimension {}.".format(embd.shape[0],
                                                       embd.shape[1]))

    embd = processingChain(embd)

    # Store both ark and scp if extention '.ark,scp' or '.scp,ark'. Or, only
    # ark if extension is '.ark'
    output_file = args.output
    if output_file.endswith('ark,scp') or output_file.endswith('scp,ark'):
        output_file = output_file.rstrip('ark,scp')
        output_file = output_file.rstrip('scp,ark')
        with kaldiio.WriteHelper('ark,scp:' + output_file + "ark," +
                                 output_file + 'scp') as writer:
            for i, u in enumerate(utt):
                e = embd[i]
                writer(u, e)

    elif output_file.endswith('ark'):
        with kaldiio.WriteHelper('ark:' + output_file) as writer:
            for i, u in enumerate(utt):
                e = embd[i]
                writer(u, e)
    else:
        raise Exception(
            "Invalid file extension of output file {}".format(output_file))

    print("Wrote {} embeddings of dimension {}.".format(
        embd.shape[0], embd.shape[1]))
