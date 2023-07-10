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

from contextlib import nullcontext
import tableprint as tp

import torch
import torchnet as tnt
from functools import partial


def run_epoch(dataloader,
              loader_size,
              model,
              criterion,
              optimizer,
              scheduler,
              epoch,
              logger,
              scaler,
              enable_amp,
              log_batch_interval=100,
              device=torch.device('cuda')):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    # https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/executor.py#L40
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_context = partial(model.join, throw_on_early_termination=True)
    else:
        model_context = nullcontext

    try:
        with model_context():
            for i, batch in enumerate(dataloader):

                cur_iter = (epoch - 1) * loader_size + i
                scheduler.step(cur_iter)

                # queries: (B, T, F)
                queries = batch['queries'].squeeze(1).float().to(device)
                # keys: (B, T, F)
                keys = batch['keys'].squeeze(1).float().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    logits, labels = model(queries, keys)
                    loss = criterion(logits, labels)

                # loss, acc
                loss_meter.add(loss.item())
                acc_meter.add(logits.cpu().detach().numpy(),
                              labels.cpu().numpy())

                # updata the model
                optimizer.zero_grad()
                # scaler does nothing here if enable_amp=False
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # log
                if (i + 1) % log_batch_interval == 0:
                    logger.info(
                        tp.row((epoch, i + 1, scheduler.get_lr()) +
                               (loss_meter.value()[0], acc_meter.value()[0]),
                               width=10,
                               style='grid'))
    except RuntimeError as e:
        if 'exhausted' in str(e):
            # Detect the error that one process has exhausted the inputs,
            # Because there is self-implemented multi-processing commutation operation in moco,
            # and the model.join() operation has no influence on such operation.
            # We mush stop all the processes when one process has exhausted the inputs.
            pass
        else:
            raise e

    logger.info(
        tp.row((epoch, i + 1, scheduler.get_lr()) +
               (loss_meter.value()[0], acc_meter.value()[0]),
               width=10,
               style='grid'))
