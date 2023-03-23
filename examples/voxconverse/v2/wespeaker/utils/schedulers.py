# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
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

import math


class MarginScheduler:

    def __init__(self,
                 model,
                 epoch_iter,
                 increase_start_epoch,
                 fix_start_epoch,
                 initial_margin,
                 final_margin,
                 update_margin,
                 increase_type='exp'):
        '''
        The margin is fixed as initial_margin before increase_start_epoch,
        between increase_start_epoch and fix_start_epoch, the margin is
        exponentially increasing from initial_margin to final_margin
        after fix_start_epoch, the margin is fixed as final_margin.
        '''
        self.model = model
        self.increase_start_iter = (increase_start_epoch - 1) * epoch_iter
        self.fix_start_iter = (fix_start_epoch - 1) * epoch_iter
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type

        self.fix_already = False
        self.current_iter = 0
        self.update_margin = update_margin and hasattr(self.model.projection,
                                                       'update')
        self.increase_iter = self.fix_start_iter - self.increase_start_iter

        self.init_margin()

    def init_margin(self):
        if hasattr(self.model.projection, 'update'):
            self.model.projection.update(margin=self.initial_margin)

    def get_increase_margin(self):
        initial_val = 1.0
        final_val = 1e-3

        current_iter = self.current_iter - self.increase_start_iter

        if self.increase_type == 'exp':  # exponentially increase the margin
            ratio = 1.0 - math.exp(
                (current_iter / self.increase_iter) *
                math.log(final_val / (initial_val + 1e-6))) * initial_val
        else:  # linearly increase the margin
            ratio = 1.0 * current_iter / self.increase_iter
        return self.initial_margin + (self.final_margin -
                                      self.initial_margin) * ratio

    def step(self, current_iter=None):
        if not self.update_margin or self.fix_already:
            return

        if current_iter is not None:
            self.current_iter = current_iter

        if self.current_iter >= self.fix_start_iter:
            self.fix_already = True
            if hasattr(self.model.projection, 'update'):
                self.model.projection.update(margin=self.final_margin)
        elif self.current_iter >= self.increase_start_iter:
            if hasattr(self.model.projection, 'update'):
                self.model.projection.update(margin=self.get_increase_margin())

        self.current_iter += 1

    def get_margin(self):
        try:
            margin = self.model.projection.margin
        except Exception:
            margin = 0.0

        return margin


class BaseClass:
    '''
    Base Class for learning rate scheduler
    '''

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 warm_from_zero=False):
        '''
        warm_up_epoch: the first warm_up_epoch is the multiprocess warm-up stage
        scale_ratio: multiplied to the current lr in the multiprocess training
        process
        '''
        self.optimizer = optimizer
        self.max_iter = num_epochs * epoch_iter
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.scale_ratio = scale_ratio
        self.current_iter = 0
        self.warm_up_iter = warm_up_epoch * epoch_iter
        self.warm_from_zero = warm_from_zero

    def get_multi_process_coeff(self):
        lr_coeff = 1.0 * self.scale_ratio
        if self.current_iter < self.warm_up_iter:
            if self.warm_from_zero:
                lr_coeff = self.scale_ratio * self.current_iter / self.warm_up_iter
            elif self.scale_ratio > 1:
                lr_coeff = (self.scale_ratio -
                            1) * self.current_iter / self.warm_up_iter + 1.0

        return lr_coeff

    def get_current_lr(self):
        '''
        This function should be implemented in the child class
        '''
        return 0.0

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

    def step(self, current_iter=None):
        if current_iter is not None:
            self.current_iter = current_iter

        self.set_lr()
        self.current_iter += 1

    def step_return_lr(self, current_iter=None):
        if current_iter is not None:
            self.current_iter = current_iter

        current_lr = self.get_current_lr()
        self.current_iter += 1

        return current_lr


class ExponentialDecrease(BaseClass):

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 warm_from_zero=False):
        super().__init__(optimizer, num_epochs, epoch_iter, initial_lr,
                         final_lr, warm_up_epoch, scale_ratio, warm_from_zero)

    def get_current_lr(self):
        lr_coeff = self.get_multi_process_coeff()
        current_lr = lr_coeff * self.initial_lr * math.exp(
            (self.current_iter / self.max_iter) *
            math.log(self.final_lr / self.initial_lr))
        return current_lr


class TriAngular2(BaseClass):
    '''
    The implementation of https://arxiv.org/pdf/1506.01186.pdf
    '''

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 cycle_step=2,
                 reduce_lr_diff_ratio=0.5):
        super().__init__(optimizer, num_epochs, epoch_iter, initial_lr,
                         final_lr, warm_up_epoch, scale_ratio)

        self.reduce_lr_diff_ratio = reduce_lr_diff_ratio
        self.cycle_iter = cycle_step * epoch_iter
        self.step_size = self.cycle_iter // 2

        self.max_lr = initial_lr
        self.min_lr = final_lr
        self.gap = self.max_lr - self.min_lr

    def get_current_lr(self):
        lr_coeff = self.get_multi_process_coeff()
        point = self.current_iter % self.cycle_iter
        cycle_index = self.current_iter // self.cycle_iter

        self.max_lr = self.min_lr + self.gap * self.reduce_lr_diff_ratio**cycle_index

        if point <= self.step_size:
            current_lr = self.min_lr + (self.max_lr -
                                        self.min_lr) * point / self.step_size
        else:
            current_lr = self.max_lr - (self.max_lr - self.min_lr) * (
                point - self.step_size) / self.step_size

        current_lr = lr_coeff * current_lr

        return current_lr


def show_lr_curve(scheduler):
    import matplotlib.pyplot as plt

    lr_list = []
    for current_lr in range(0, scheduler.max_iter):
        lr_list.append(scheduler.step_return_lr(current_lr))
    data_index = list(range(1, len(lr_list) + 1))

    plt.plot(data_index, lr_list, '-o', markersize=1)
    plt.legend(loc='best')
    plt.xlabel("Iteration")
    plt.ylabel("LR")

    plt.show()


if __name__ == '__main__':
    optimizer = None
    num_epochs = 6
    epoch_iter = 500
    initial_lr = 0.6
    final_lr = 0.1
    warm_up_epoch = 2
    scale_ratio = 4
    scheduler = ExponentialDecrease(optimizer, num_epochs, epoch_iter,
                                    initial_lr, final_lr, warm_up_epoch,
                                    scale_ratio)
    # scheduler = TriAngular2(optimizer,
    #                         num_epochs,
    #                         epoch_iter,
    #                         initial_lr,
    #                         final_lr,
    #                         warm_up_epoch,
    #                         scale_ratio,
    #                         cycle_step=2,
    #                         reduce_lr_diff_ratio=0.5)

    show_lr_curve(scheduler)
