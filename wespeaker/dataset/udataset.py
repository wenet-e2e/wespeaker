# Copyright (c) 2022 Horizon Robtics. (authors: Binbin Zhang)
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

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from wespeaker.utils.file_utils import read_lists, read_scp
from wespeaker.dataset.lmdb_data import LmdbData
import wespeaker.dataset.processor as processor


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def Dataset(data_list_file,
            spk2id_file,
            conf,
            reverb_lmdb_file=None,
            noise_lmdb_file=None):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_list_file: shard list file
            spk2id_file: speaker to id file
            reverb_lmdb_file: reverb data source lmdb file
            noise_lmdb_file: noise data source lmdb file
    """
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', False)
    # Global shuffle
    dataset = DataList(lists, shuffle=shuffle)
    dataset = Processor(dataset, processor.url_opener)
    dataset = Processor(dataset, processor.tar_file_and_group)
    # Local shuffle
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    spk2id = read_scp(spk2id_file)
    spk2id = {x[0]: int(x[1]) for x in spk2id}
    dataset = Processor(dataset, processor.speaker_to_id, spk2id)

    speed_perturb_flag = conf.get('speed_perturb', False)
    if speed_perturb_flag:
        dataset = Processor(dataset, processor.speed_perturb, len(spk2id))

    # random chunk
    dataset = Processor(dataset, processor.random_chunk, 2.0)

    # Optional add reverb
    if reverb_lmdb_file is not None:
        reverb_data = LmdbData(reverb_lmdb_file)
        dataset = Processor(dataset, processor.add_reverb, reverb_data)

    # Optional add noise
    if noise_lmdb_file:
        noise_data = LmdbData(noise_lmdb_file)
        dataset = Processor(dataset, processor.add_noise, noise_data)

    fbank_conf = conf.get('fbank_conf', {})
    dataset = Processor(dataset, processor.compute_fbank, **fbank_conf)

    spec_aug_flag = conf.get('spec_aug', True)
    if spec_aug_flag:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)

    return dataset
