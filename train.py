# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

import json
import os
import re
from pprint import pformat

import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import *
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.executor import run_epoch
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id


def create_wespeaker_list(scp_file, utt2spk_file, output_file):
    utt2wav = {}
    with open(scp_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            utt2wav[parts[0]] = parts[1]

    with open(utt2spk_file, 'r') as f, open(output_file, 'w') as out:
        for line in f:
            utt, spk = line.strip().split()
            # This JSON format is what WeSpeaker's parse_raw expects
            data = {"key": utt, "wav": utt2wav[utt], "spk": spk}
            out.write(json.dumps(data) + '\n')



def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    model_dir = "finetuning_models"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(configs['exp_dir'], exist_ok=True)
    logger = get_logger(configs['exp_dir'], 'train.log')
    # seed
    set_seed(configs['seed'])
    rank = 0

    # train data
    create_wespeaker_list('dataset/train_wav.scp', 'dataset/train_utt2spk', 'dataset/train.list')
    create_wespeaker_list('dataset/dev_wav.scp', 'dataset/dev_utt2spk', 'dataset/dev.list')

    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)

    logger.info("<== Data statistics ==>")
    logger.info("train data num: {}, spk num: {}".format(
        len(train_utt_spk_list), len(spk2id_dict)))

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None))
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'])
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
    epoch_iter = sample_num_per_epoch // batch_size
    logger.info("<== Dataloaders ==>")
    logger.info("train dataloaders created")
    logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model: frontend (optional) => speaker model => projection layer
    logger.info("<== Model ==>")
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    num_params = sum(param.numel() for param in model.parameters())
    logger.info('speaker_model size: {}'.format(num_params))
    # For model_init, only frontend and speaker model are needed !!!
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')
    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args'][
        'embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    load_checkpoint(model, configs['model_init'])

    # print model
    for line in pformat(model).split('\n'):
        logger.info(line)

    # If specify checkpoint, load some info from checkpoint.
    # For checkpoint, frontend, speaker model, and projection layer
    # are all needed !!!
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()
    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    if 'initial_lr' in configs['scheduler_args']:
        configs['optimizer_args']['lr'] = (
            configs['scheduler_args']['initial_lr']
        )
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    logger.info("<========== Training process ==========>")
    header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
    for line in tp.header(header, width=10, style='grid').split('\n'):
        logger.info(line)

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                  epoch_iter,
                  model,
                  criterion,
                  optimizer,
                  scheduler,
                  margin_scheduler,
                  epoch,
                  logger,
                  scaler,
                  device=device,
                  configs=configs)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch > configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))


if __name__ == '__main__':
    fire.Fire(train)
