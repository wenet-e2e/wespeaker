#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang

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
from wespeaker.models.projections import get_projection
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, spk2id
from wespeaker.utils.file_utils import read_scp
from wespeaker.utils.executor_uio import run_epoch
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.dataset.udataset import Dataset

def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    # dist configs
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][rank])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print(model_dir + " already exists !!!")
            if checkpoint is None:
                exit(1)
    dist.barrier()  # let the rank 0 mkdir first

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

    # wav/feat
    train_scp = configs['dataset_conf']['train_scp']
    train_label = configs['dataset_conf']['train_label']
    train_data_list = read_scp(train_scp)
    if rank == 0:
        logger.info("<== Feature ==>")
        logger.info("train wav/feat num: {}".format(len(train_data_list)))

    # spk label
    train_utt_spk_list = read_scp(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    train_utt2spkid_dict = {
        utt_spk[0]: spk2id_dict[utt_spk[1]]
        for utt_spk in train_utt_spk_list
    }
    if rank == 0:
        logger.info("<== Labels ==>")
        logger.info("train label num: {}, spk num: {}".format(
            len(train_utt2spkid_dict), len(spk2id_dict)))

    train_dataset = Dataset(
        configs.get('data_type', 'raw'),
        configs['train_list'],
        configs['spk2id'],
        configs['dataset_conf'],
        reverb_lmdb_file=configs.get('reverb_lmdb', None),
        noise_lmdb_file=configs.get('noise_lmdb', None)
    )
    train_dataloader = DataLoader(train_dataset,
                                  **configs['dataloader_args'])
    batchsize = configs['dataloader_args']['batch_size']
    lenloader = len(train_data_list) // batchsize // world_size
    logger.info('word_size: {}'.format(world_size))
    logger.info('lenloader: {}'.format(lenloader))


    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")

    # model
    logger.info("<== Model ==>")
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    if configs['model_init'] is not None:
        logger.info('Load intial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    else:
        logger.info('Train model from scratch...')
    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args']['embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    if configs['feature_args']['raw_wav'] and configs['dataset_conf']['speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(model_dir, 'init.zip'))

    # If specify checkpoint, load some info from checkpoint.
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer = getattr(torch.optim, configs['optimizer'])(
        ddp_model.parameters(), **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    print(world_size)
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = lenloader
    configs['scheduler_args']['process_num'] = world_size
    scheduler = getattr(schedulers, configs['scheduler'])(
        optimizer, **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = lenloader
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

    # training
    dist.barrier()  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier()  # synchronize here

    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        # train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                  lenloader,
                  ddp_model,
                  criterion,
                  optimizer,
                  scheduler,
                  margin_scheduler,
                  epoch,
                  logger,
                  log_batch_interval=configs['log_batch_interval'],
                  device=device)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch >= configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model,
                    os.path.join(model_dir, 'model_{}.pt'.format(epoch)))

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))


if __name__ == '__main__':
    fire.Fire(train)
