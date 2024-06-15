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
    parser.add_argument('--in_path',
                        type=str,
                        default='',
                        help='Path where to load original processing chain.')
    parser.add_argument('--out_path',
                        type=str,
                        default='',
                        help='Path where to save updated processing chain.')
    parser.add_argument('--link_no_to_remove',
                        type=int,
                        default='',
                        help='Input scp file.')
    parser.add_argument(
        '--new_link',
        type=str,
        default='',
        help='new link, e.g., "mean-subtract --scp new_scp_for_mean.scp".')
    args = parser.parse_args()

    processingChain = EmbeddingProcessingChain()
    processingChain.load(args.in_path)
    processingChain.update_link(args.link_no_to_remove, args.new_link)
    processingChain.save(args.out_path)
