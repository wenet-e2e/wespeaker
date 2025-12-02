# tools/merge_lora.py

import torch
import fire

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.frontend import frontend_class_dict
from wespeaker.utils.utils import parse_config_or_kwargs
from peft import PeftModel


def merge(config: str, checkpoint_in: str, checkpoint_out: str) -> None:
    print(f"Loading Stage 1 config from: {config}")
    configs = parse_config_or_kwargs(config)

    if (
        "dataset_args" not in configs
        or "frontend" not in configs["dataset_args"]
    ):
        print(
            "Error: 'dataset_args.frontend' not found in config file "
            f"{config}."
        )
        return

    frontend_name = configs["dataset_args"]["frontend"]
    frontend_args_key = f"{frontend_name}_args"

    if frontend_args_key not in configs["dataset_args"]:
        print(
            "Error: 'dataset_args.{frontend_args_key}' not found in config "
            f"file {config}."
        )
        return

    frontend_args = configs["dataset_args"][frontend_args_key]

    if not frontend_args.get("use_lora", False):
        print(
            "Warning: 'use_lora: True' is not set in "
            f"'dataset_args.{frontend_args_key}' of config {config}."
        )
        print("Make sure you are using the Stage 1 (LoRA) config file.")

    print(f"Building frontend: {frontend_name}")
    frontend_class = frontend_class_dict[frontend_name]
    frontend = frontend_class(**frontend_args)

    print(f"Building model: {configs['model']}")
    model_class = get_speaker_model(configs["model"])
    model = model_class(**configs["model_args"])

    model.frontend = frontend

    print(f"Loading LoRA checkpoint from: {checkpoint_in}")
    checkpoint = torch.load(checkpoint_in, map_location="cpu")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint, strict=False)

    print("Merging LoRA weights into the base model...")
    encoder = getattr(getattr(model, "frontend", None), "encoder", None)

    if isinstance(encoder, PeftModel):
        model.frontend.encoder = encoder.merge_and_unload()
        print("After merge, encoder type:", type(model.frontend.encoder))
        print("Merge successful.")
    else:
        print("Error: no PeftModel found to merge.")
        print(
            "  - Check if model.frontend.encoder is an instance of "
            "PeftModel"
        )
        print(f"  - model.frontend type: {type(model.frontend)}")
        if hasattr(model, "frontend") and hasattr(model.frontend, "encoder"):
            print(
                "  - model.frontend.encoder type: "
                f"{type(model.frontend.encoder)}"
            )
        return

    if "projection_args" in configs:
        try:
            from wespeaker.models.projections import get_projection

            print("Building projection layer...")
            projection = get_projection(configs["projection_args"])
            model.add_module("projection", projection)

            for key in ("projection.weight", "projection.bias"):
                if key in checkpoint:
                    print(f"Loading {key} from checkpoint.")
                    model.state_dict()[key].copy_(checkpoint[key])
                else:
                    print(f"Warning: {key} not found in LoRA checkpoint.")
        except Exception as exc:
            print(
                "Warning: failed to build or load projection: "
                f"{exc}"
            )

    print(f"Saving merged model to: {checkpoint_out}")
    torch.save(model.state_dict(), checkpoint_out)
    print("Merged model saved successfully.")


if __name__ == "__main__":
    fire.Fire(merge)
