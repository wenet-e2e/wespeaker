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

import torch

from wespeaker.utils.file_utils import read_lists
from wespeaker.dataset.lmdb_data import LmdbData
import wespeaker.dataset.processor as processor
import wespeaker.ssl.dataset.processor as ssl_processor
from wespeaker.dataset.dataset import Processor, DataList


def dino_collate_fn(batch):
    key_list, label_list = [], []
    local_chunks_list, global_chunks_list = [], []

    for sample_dict in batch:
        key_list.append(sample_dict['key'])
        label_list.append(sample_dict['label'])
        local_chunks_list.append(
            torch.stack(sample_dict['feat']['local_chunks']))
        global_chunks_list.append(
            torch.stack(sample_dict['feat']['global_chunks']))

    return dict(
        key=key_list,
        label=label_list,
        local_chunks=torch.stack(local_chunks_list),
        global_chunks=torch.stack(global_chunks_list),
    )


def contrastive_collate_fn(batch):
    local_chunks_list, global_chunks_list = [], []

    for sample_dict in batch:
        local_chunks_list.append(
            torch.stack(sample_dict['feat']['local_chunks']))
        global_chunks_list.append(
            torch.stack(sample_dict['feat']['global_chunks']))

    return dict(
        keys=torch.stack(local_chunks_list),
        queries=torch.stack(global_chunks_list),
    )


def SSLDataset(data_type,
               data_list_file,
               configs,
               spk2id_dict,
               whole_utt=False,
               reverb_lmdb_file=None,
               noise_lmdb_file=None):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw/feat file level. The second is local shuffle
        at training samples level.

        Args:
            data_type(str): shard/raw/feat
            data_list_file: data list file
            configs: dataset configs
            spk2id_dict: spk2id dict
            reverb_lmdb_file: reverb data source lmdb file
            noise_lmdb_file: noise data source lmdb file
            whole_utt: use whole utt or random chunk
    """
    assert data_type in ['shard', 'raw', 'feat']
    lists = read_lists(data_list_file)
    shuffle = configs.get('shuffle', False)
    # Global shuffle
    dataset = DataList(lists, shuffle=shuffle)
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    elif data_type == 'raw':
        dataset = Processor(dataset, processor.parse_raw)
    else:
        dataset = Processor(dataset, processor.parse_feat)

    if configs.get('filter', True):
        # Filter the data with unwanted length
        filter_conf = configs.get('filter_args', {})
        dataset = Processor(dataset,
                            processor.filter,
                            frame_shift=configs['fbank_args'].get(
                                'frame_shift', 10),
                            data_type=data_type,
                            **filter_conf)

    # Local shuffle
    if shuffle:
        dataset = Processor(dataset, processor.shuffle,
                            **configs['shuffle_args'])

    # spk2id
    dataset = Processor(dataset, ssl_processor.spk_to_id, spk2id_dict)

    if data_type == 'feat':
        if not whole_utt:
            # random chunk
            chunk_info_args = configs['chunk_info_args']
            chunk_info_args['data_type'] = 'feat'
            dataset = Processor(dataset, ssl_processor.random_chunk_for_dino,
                                **chunk_info_args)
    else:
        # resample
        resample_rate = configs.get('resample_rate', 16000)
        dataset = Processor(dataset, processor.resample, resample_rate)
        # speed perturb
        speed_perturb_flag = configs.get('speed_perturb', True)
        if speed_perturb_flag:
            spk_num = len(spk2id_dict) if spk2id_dict is not None else 0
            dataset = Processor(dataset, processor.speed_perturb, spk_num)
        if not whole_utt:
            # random chunk
            frame_shift = configs['fbank_args'].get('frame_shift', 10)
            frame_length = configs['fbank_args'].get('frame_length', 25)
            chunk_info_args = configs['chunk_info_args']
            for key in chunk_info_args:
                if 'chunk_len' in key:
                    chunk_info_args[key] = (
                        (chunk_info_args[key] - 1) * frame_shift +
                        frame_length) * resample_rate // 1000
            chunk_info_args['data_type'] = data_type
            dataset = Processor(dataset, ssl_processor.random_chunk_for_dino,
                                **chunk_info_args)
        # add reverb & noise
        aug_prob = configs.get('aug_prob', 0.6)
        if (reverb_lmdb_file and noise_lmdb_file) and (aug_prob > 0.0):
            reverb_data = LmdbData(reverb_lmdb_file)
            noise_data = LmdbData(noise_lmdb_file)
            dataset = Processor(dataset, ssl_processor.add_reverb_noise,
                                reverb_data, noise_data, resample_rate,
                                aug_prob)
        # compute fbank
        dataset = Processor(dataset, ssl_processor.compute_fbank,
                            **configs['fbank_args'])

    # apply cmvn
    dataset = Processor(dataset, ssl_processor.apply_cmvn)

    # spec augmentation
    spec_aug_flag = configs.get('spec_aug', True)
    if spec_aug_flag:
        dataset = Processor(dataset, ssl_processor.spec_aug,
                            **configs['spec_aug_args'])

    return dataset
