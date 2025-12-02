# Copyright (c) 2025 Qituan Shangguan (2369144677@qq.com)
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

import os
import json
import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2BertModel,
    Wav2Vec2BertConfig,
    BitsAndBytesConfig,
    AutoFeatureExtractor,
    AutoModel,
)
from peft import LoraConfig, get_peft_model


def create_bnb_config(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="llm_int8",
    bnb_8bit_compute_dtype="bfloat16",
):
    # Note: llm_int8 might not be the optimal quant_type for audio models.
    # Consider "int8" or potentially "nf4" if available and suitable.
    # Also, compute_dtype might need adjustment based on hardware capability.
    return BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        bnb_8bit_use_double_quant=bnb_8bit_use_double_quant,
        bnb_8bit_quant_type=bnb_8bit_quant_type,
        bnb_8bit_compute_dtype=getattr(torch, bnb_8bit_compute_dtype),
    )


def create_lora_config(
    model_type="w2v-bert",
    r=16,
    lora_alpha=32,
    target_modules=None,
    lora_dropout=0.1,
    bias="none",
):
    """LoRA config specifically for w2v-bert."""
    if model_type != "w2v-bert":
        raise ValueError(
            "This function is specifically for w2v-bert, got "
            f"{model_type}"
        )

    if target_modules is None:
        # Common targets for Wav2Vec2Bert.
        target_modules = ["linear_q", "linear_v"]

    # Task type for feature extraction is suitable here.
    task_type = "FEATURE_EXTRACTION"

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )


class W2VBertFrontend(nn.Module):
    """
    Wav2Vec2-BERT Frontend for wespeaker, potentially with LoRA.
    """

    PEFT_INDICATORS = ["lora_", "adapter", "prefix", "prompt"]

    def __init__(
        self,
        model_name="facebook/w2v-bert-2.0",
        download_dir="./w2vbert_hub",
        frozen=True,
        use_bnb=False,
        bnb_config_args=None,
        use_lora=False,
        lora_config_args=None,
        model_config_file=None,
        sample_rate=16000,
    ):
        super().__init__()
        self.model_type = "w2v-bert"
        self.model_config_file = model_config_file
        self.frozen_encoder = frozen
        self.use_lora = use_lora
        self.sr = sample_rate

        if not frozen and use_bnb:
            raise ValueError(
                "Full fine-tuning (frozen=False) and quantization "
                "(use_bnb=True) are not supported simultaneously."
            )

        if os.path.isdir(model_name):
            local_model_path = model_name
        else:
            local_model_path = model_name

        try:
            self.processor = AutoFeatureExtractor.from_pretrained(
                local_model_path,
                local_files_only=os.path.isdir(local_model_path),
            )
        except Exception as e:
            raise IOError(
                "Failed to load feature extractor from "
                f"{local_model_path}: {e}"
            ) from e

        bnb_config = (
            create_bnb_config(**(bnb_config_args or {})) if use_bnb else None
        )
        self._setup_model(local_model_path, bnb_config)

        if use_lora:
            lora_config = create_lora_config(
                model_type=self.model_type,
                **(lora_config_args or {}),
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            print("LoRA applied to the model.")
            if frozen:
                print(
                    "Note: Encoder was initially frozen, but LoRA layers are "
                    "now trainable."
                )
                self.frozen_encoder = True

        if frozen:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()

    def forward(self, wavs: torch.Tensor, wavs_len: torch.Tensor):
        wavs_cpu = wavs.cpu()
        wavs_list = list(wavs_cpu)

        features = self.processor(
            wavs_list,
            return_tensors="pt",
            sampling_rate=self.sr,
            padding="longest",
        )
        features = features.to(wavs.device)

        # Extract the required tensors from the features dictionary.
        input_tensor = None
        if "input_values" in features:
            input_tensor = features["input_values"]
        elif "input_features" in features:
            input_tensor = features["input_features"]
        elif "input_ids" in features:
            input_tensor = features["input_ids"]
        else:
            raise ValueError(
                "Could not find audio data in the processor output."
            )

        attention_mask_tensor = features.get("attention_mask")

        if self.use_lora:
            # For Stage 1 (LoRA training), the model is a PeftModel wrapper.
            # We need to access the underlying model.
            outputs = self.encoder.base_model.model(
                input_tensor,
                attention_mask=attention_mask_tensor,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            # For Stage 2/3 (full fine-tuning) we call it directly.
            outputs = self.encoder(
                input_tensor,
                attention_mask=attention_mask_tensor,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden_state = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states

        return all_hidden_states, last_hidden_state

    def _setup_model(self, model_path, bnb_config):
        """Loads the Wav2Vec2BertModel."""
        try:
            config_file_path = None
            if (
                self.model_config_file is not None
                and os.path.isfile(
                    os.path.join(model_path, self.model_config_file)
                )
            ):
                config_file_path = os.path.join(
                    model_path,
                    self.model_config_file,
                )

            if config_file_path is not None:
                print(
                    "Loading model with custom config: "
                    f"{self.model_config_file}"
                )
                with open(config_file_path, "r") as f:
                    config_dict = json.load(f)
                config = Wav2Vec2BertConfig(**config_dict)
                full_model = Wav2Vec2BertModel(config)

                # Load weights, potentially handling prune config naming.
                ckpt_file = os.path.join(
                    model_path,
                    "model.safetensors",
                )
                if not os.path.exists(ckpt_file):
                    ckpt_file = os.path.join(
                        model_path,
                        "pytorch_model.bin",
                    )

                if ckpt_file.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    ckpt_state_dict = load_file(ckpt_file, device="cpu")
                else:
                    ckpt_state_dict = torch.load(
                        ckpt_file,
                        map_location="cpu",
                    )

                cur_state_dict = full_model.state_dict()
                missing_keys, unexpected_keys = [], []
                for k in cur_state_dict.keys():
                    # Skip potential pruning params.
                    if "hard_concrete" in k:
                        continue
                    if k in ckpt_state_dict:
                        if (
                            cur_state_dict[k].shape
                            == ckpt_state_dict[k].shape
                        ):
                            cur_state_dict[k] = ckpt_state_dict[k]
                        else:
                            print(
                                "Shape mismatch for key "
                                f"{k}: model needs "
                                f"{cur_state_dict[k].shape}, checkpoint "
                                f"has {ckpt_state_dict[k].shape}"
                            )
                            # Treat shape mismatch as missing for
                            # load_state_dict.
                            missing_keys.append(k)
                    else:
                        missing_keys.append(k)

                # Collect unexpected keys from checkpoint.
                for k in ckpt_state_dict.keys():
                    if k not in cur_state_dict:
                        unexpected_keys.append(k)

                load_result = full_model.load_state_dict(
                    cur_state_dict,
                    strict=False,
                )
                print(
                    "Custom config model loaded. Missing: "
                    f"{load_result.missing_keys}, unexpected: "
                    f"{load_result.unexpected_keys}"
                )
            else:
                print(f"Loading model using AutoModel from: {model_path}")
                full_model = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=os.path.isdir(model_path),
                    quantization_config=bnb_config,
                )

            self.encoder = full_model
            self.d_model = self.encoder.config.hidden_size
            # W2VBERT does not have a direct equivalent to whisper's
            # encoder_layers+1 for hidden states count in config.
            # It depends on output_hidden_states=True during forward pass.
            # We can store the config value for reference.
            self.n_config_layers = self.encoder.config.num_hidden_layers

            # Remove the masked_spec_embed parameter that is not needed
            # during fine-tuning.
            if hasattr(self.encoder, "masked_spec_embed"):
                delattr(self.encoder, "masked_spec_embed")
                print("'masked_spec_embed' attribute removed.")
        except Exception as e:
            raise IOError(f"Failed to load model from {model_path}:{e}") from e

    def _is_peft_parameter(self, param_name):
        return any(
            indicator in param_name for indicator in self.PEFT_INDICATORS
        )

    def _module_has_peft_parameter(self, module):
        if hasattr(module, "named_parameters"):
            for param_name, _ in module.named_parameters():
                if self._is_peft_parameter(param_name):
                    return True
        return False

    def freeze_encoder(self):
        self.frozen_encoder = True
        if hasattr(self, "encoder"):
            print("Freezing base model parameters...")
            for name, param in self.encoder.named_parameters():
                if not self._is_peft_parameter(name):
                    param.requires_grad = False
        else:
            print("Encoder not initialized yet, cannot freeze.")

    def unfreeze_encoder(self):
        # If LoRA is used, unfreezing the whole encoder might not be intended.
        # This method will make all non-LoRA params trainable.
        self.frozen_encoder = False
        if hasattr(self, "encoder"):
            print("Unfreezing base model parameters...")
            for name, param in self.encoder.named_parameters():
                if not self._is_peft_parameter(name):
                    param.requires_grad = True
        else:
            print("Encoder not initialized yet, cannot unfreeze.")

    def train(self, mode=True):
        # Set the top-level module training mode.
        super().train(mode)
        if hasattr(self, "encoder"):
            if self.frozen_encoder and self.use_lora:
                # In frozen mode with LoRA: base model in eval,
                # LoRA layers in train.
                print(
                    "Setting train mode: base model in eval, "
                    "LoRA modules in train."
                )
                self.encoder.eval()
                for name, module in self.encoder.named_modules():
                    # Check if the module itself is a LoRA layer or
                    # contains LoRA params.
                    if (
                        "lora_" in name.lower()
                        or self._module_has_peft_parameter(module)
                    ):
                        module.train(mode)
            elif self.frozen_encoder:
                # Frozen mode without LoRA: everything in eval.
                print("Setting train mode: entire encoder in eval (frozen).")
                self.encoder.eval()
            else:
                # Not frozen: normal training for the entire encoder.
                print(
                    "Setting train mode: entire encoder in "
                    f"{'train' if mode else 'eval'}."
                )
                self.encoder.train(mode)
        return self

    def output_size(self) -> int:
        """Return the dimension of the output embedding."""
        # For adapter MFA, the input dim is d_model, but the final output dim
        # depends on the adapter and pooling layers defined outside this
        # frontend. This frontend should return the hidden dimension of the
        # base model.
        if not hasattr(self, "d_model"):
            # Should be set during _setup_model.
            raise AttributeError("Model dimension 'd_model' not set.")
        # Output dim of each hidden state from W2V-BERT.
        return self.d_model
