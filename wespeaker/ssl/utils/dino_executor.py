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

import tableprint as tp

import torch
import torchnet as tnt
from wespeaker.ssl.utils.dino_utils import (
    cancel_gradients_last_layer,
    clip_gradients,
)


def run_epoch(dataloader,
              epoch_iter,
              model,
              optimizer,
              lr_schedule,
              wd_schedule,
              mt_schedule,
              epoch,
              logger,
              scaler,
              enable_amp,
              clip_grad=3.0,
              freeze_last_layer=1,
              log_batch_interval=100,
              device=torch.device('cuda')):
    model.train()

    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()

    for i, batch in enumerate(dataloader):

        cur_iter = (epoch - 1) * epoch_iter + i
        # --------------- Update dynamic hyper-parameter ---------------
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[cur_iter]
            if k == 0:
                param_group['weight_decay'] = wd_schedule[cur_iter]
        # --------------- Update dynamic hyper-parameter ---------------

        # (B, chunk_num, T, F)
        local_feats = batch['local_chunks'].float().to(device)
        # (B, chunk_num', T, F)
        global_feats = batch['global_chunks'].float().to(device)

        # (B, chunk_num, T, F) --> (chunk_num, B, T, F)
        # --> (chunk_num * B, T, F)
        local_T, local_F = local_feats.shape[-2:]
        local_feats = local_feats.transpose(0, 1).contiguous().view(
            -1, local_T, local_F)
        global_T, global_F = global_feats.shape[-2:]
        global_feats = global_feats.transpose(0, 1).contiguous().view(
            -1, global_T, global_F)

        with torch.cuda.amp.autocast(enabled=enable_amp):
            loss = model(local_feats, global_feats, epoch - 1)

        # loss, acc
        loss_meter.add(loss.item())

        # update the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        clip_gradients(model, clip_grad)
        cancel_gradients_last_layer(epoch - 1, model.module.s_model,
                                    freeze_last_layer)

        scaler.step(optimizer)
        scaler.update()

        # EMA update for teacher
        m = mt_schedule[cur_iter]  # momentum parameter
        model.module.ema_update(m)

        # log
        if (i + 1) % log_batch_interval == 0:
            logger.info(
                tp.row((epoch, i + 1, lr_schedule[cur_iter],
                        loss_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row((epoch, i + 1, lr_schedule[cur_iter], loss_meter.value()[0]),
               width=10,
               style='grid'))
