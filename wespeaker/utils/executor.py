#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang

import tableprint as tp

import torch
import torchnet as tnt


def run_epoch(dataloader,
              model,
              criterion,
              optimizer,
              scheduler,
              margin_scheduler,
              epoch,
              logger,
              log_batch_interval=100,
              device=torch.device('cuda')):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    with torch.set_grad_enabled(True):
        for i, (utts, features, targets) in enumerate(dataloader):

            cur_iter = (epoch - 1) * len(dataloader) + i
            scheduler.step(cur_iter)
            margin_scheduler.step(cur_iter)

            features = features.float().to(device)  # (B,T,F)
            targets = targets.long().to(device)
            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)

            loss = criterion(outputs, targets)
            # loss, acc
            loss_meter.add(loss.item())
            acc_meter.add(outputs.cpu().detach().numpy(),
                          targets.cpu().numpy())

            # updata the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            if i % log_batch_interval == 0:
                logger.info(
                    tp.row((epoch, i, scheduler.get_lr(),
                            margin_scheduler.get_margin()) +
                           (loss_meter.value()[0], acc_meter.value()[0]),
                           width=10,
                           style='grid'))

    logger.info(
        tp.row((epoch, len(dataloader), scheduler.get_lr(),
                margin_scheduler.get_margin()) +
               (loss_meter.value()[0], acc_meter.value()[0]),
               width=10,
               style='grid'))
