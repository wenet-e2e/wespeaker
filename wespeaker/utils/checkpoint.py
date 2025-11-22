# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#               2021 Hongji Wang (jijijiang77@gmail.com)
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
import logging


def load_checkpoint(model: torch.nn.Module, path: str):
    """
    Load a checkpoint and handle potential size mismatch in
    the projection layer.
    """
    checkpoint = torch.load(path, map_location="cpu")
    current_state_dict = model.state_dict()

    proj_key = "projection.weight"
    if proj_key in checkpoint and proj_key in current_state_dict:
        ckpt_w = checkpoint[proj_key]
        curr_w = current_state_dict[proj_key]

        # Check if shapes mismatch
        if ckpt_w.shape != curr_w.shape:
            logging.warning(
                f"Size mismatch for {proj_key}: "
                f"checkpoint has shape {ckpt_w.shape}, "
                f"current model has shape {curr_w.shape}."
            )

            ckpt_len = ckpt_w.shape[0]
            curr_len = curr_w.shape[0]

            # Case: checkpoint from speed-perturbed training
            # (num_classes * 3) to LMFT training (original num_classes)
            if ckpt_len > curr_len:
                logging.info(
                    "Loading the first %d rows from checkpoint's "
                    "projection layer.",
                    curr_len,
                )
                # Only use the first part of weights from checkpoint
                checkpoint[proj_key] = ckpt_w[:curr_len, :]

                # Also handle bias if present
                bias_key = "projection.bias"
                if bias_key in checkpoint and bias_key in current_state_dict:
                    ckpt_b = checkpoint[bias_key]
                    if ckpt_b.shape[0] > curr_len:
                        checkpoint[bias_key] = ckpt_b[:curr_len]

    # Load with strict=False to tolerate missing / extra keys
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint,
        strict=False,
    )

    # Filter out projection keys we already handled explicitly so logs
    # focus on truly unexpected tensors.
    final_unexpected_keys = [
        k for k in unexpected_keys if "projection" not in k
    ]

    for key in missing_keys:
        # Missing projection keys are expected if the source model did
        # not have projection; do not warn for those.
        if "projection" not in key:
            logging.warning("missing tensor: %s", key)

    for key in final_unexpected_keys:
        logging.warning("unexpected tensor: %s", key)



def save_checkpoint(model: torch.nn.Module, path: str):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
