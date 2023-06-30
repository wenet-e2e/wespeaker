# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
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

from contextlib import nullcontext
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

    # https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/executor.py#L40
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_context = model.join
    else:
        model_context = nullcontext

    with torch.set_grad_enabled(True), model_context():
        for i, (utts, features, targets) in enumerate(dataloader):

            cur_iter = (epoch - 1) * len(dataloader) + i
            scheduler.step(cur_iter)
            margin_scheduler.step(cur_iter)

            features = features.float().to(device)  # (B,T,F)
            targets = targets.long().to(device)
            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
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
            if (i + 1) % log_batch_interval == 0:
                logger.info(
                    tp.row((epoch, i + 1, scheduler.get_lr(),
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
