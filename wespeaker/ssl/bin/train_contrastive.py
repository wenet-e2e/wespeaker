# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
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
from pprint import pformat
import fire
import yaml
import tableprint as tp
import re

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import wespeaker.utils.schedulers as schedulers
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.ssl.models.moco_wrapper import MoCo
from wespeaker.ssl.models.simclr_wrapper import SimCLR
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed
from wespeaker.ssl.utils.contrastive_executor import run_epoch
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.ssl.dataset.dataset import SSLDataset, contrastive_collate_fn


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    # dist configs
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print(model_dir + " already exists !!!")
            if checkpoint is None:
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # train data
    with open(configs['wav_scp'], 'r') as f:
        lines = f.readlines()
        data_num = len(lines)
        del lines

    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}".format(data_num))

    # dataset and dataloader
    train_dataset = SSLDataset(configs['data_type'],
                               configs['train_data'],
                               configs['dataset_args'],
                               None,
                               reverb_lmdb_file=configs.get(
                                   'reverb_data', None),
                               noise_lmdb_file=configs.get('noise_data', None))
    train_dataloader = DataLoader(train_dataset,
                                  **configs['dataloader_args'],
                                  collate_fn=contrastive_collate_fn)
    batch_size = configs['dataloader_args']['batch_size']
    loader_size = data_num // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('loader size: {}'.format(loader_size))

    # model
    logger.info("<== Model ==>")
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    num_params = sum(param.numel() for param in model.parameters())
    if rank == 0:
        logger.info('speaker_model size: {}'.format(num_params))
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')

    if rank == 0:
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(model_dir, 'init.zip'))

    if configs['contrastive_type'] == "simclr":
        configs['simclr_args']['embed_dim'] = configs['model_args'][
            'embed_dim']
        model = SimCLR(model, **configs['simclr_args'])
    else:
        configs['moco_args']['embed_dim'] = configs['model_args']['embed_dim']
        model = MoCo(model, **configs['moco_args'])

    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)

    # If specify checkpoint, load some info from checkpoint.
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
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    device = torch.device("cuda")

    criterion = torch.nn.CrossEntropyLoss()
    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = loader_size
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    # scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
        run_epoch(train_dataloader,
                  loader_size,
                  ddp_model,
                  criterion,
                  optimizer,
                  scheduler,
                  epoch,
                  logger,
                  scaler,
                  enable_amp=configs['enable_amp'],
                  log_batch_interval=configs['log_batch_interval'],
                  device=device)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch >= configs[
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
