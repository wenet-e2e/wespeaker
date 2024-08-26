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

from wespeaker.utils.embedding_processing import EmbeddingProcessingChain

if __name__ == '__main__':
    """
    xxx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain',
                        type=str,
                        default='whitening | length-norm ',
                        help='')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    processingChain = EmbeddingProcessingChain(chain=args.chain)
    processingChain.save(args.path)
