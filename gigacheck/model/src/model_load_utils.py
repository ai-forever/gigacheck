import torch
from peft import LoraConfig, get_peft_model
from loguru import logger
import torch.nn as nn


def cast_detr_to_fp32(model):
    for name, param in model.named_parameters():
        if "detr" in name:
            param.data = param.data.to(torch.float32)


def load_lora_ckpt(lora_path, model):
    peft_config = LoraConfig.from_pretrained(lora_path)
    peft_config.target_modules = list(peft_config.target_modules)

    model = get_peft_model(model, peft_config)
    for adapter_name in model.peft_config.keys():
        load_result = model.load_adapter(lora_path, adapter_name, torch_device=str(model.device))
        missing_keys, unexpected_keys = tuple(load_result)
        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys {unexpected_keys}")
        if missing_keys:
            raise RuntimeError(f"Missing keys {missing_keys}")
    return model


def load_detr_pt(model: torch.nn.Module, ckpt_path: str):
    device = next(model.detr.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device)
    detr_state_dict = ckpt["detr"] if "detr" in ckpt else ckpt
    model.detr.load_state_dict(detr_state_dict, strict=True)
    if "criterion" in ckpt and model.criterion is not None:
        model.criterion.load_state_dict(ckpt["criterion"])

    dtype = next(model.detr.parameters()).dtype
    logger.info(f"Loaded detr from {ckpt_path}, {dtype=}")


def custom_prepare_model_for_training(
    model, output_embedding_layer_name=["lm_head"], layer_norm_names=["layer_norm"]
):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    cast_out_layers_to_float32(model, output_embedding_layer_name)
    model.gradient_checkpointing_enable()

    return model


def get_module(model, name):
    print(f"get module {name}")
    if "." in name and len(name.split(".")) == 2:
        module1, module2 = name.split(".")
        parent_module = getattr(model, module1, None)
        if parent_module is None:
            return None
        return getattr(parent_module, module2, None)
    return getattr(model, name, None)


def cast_out_layers_to_float32(model, output_embedding_layer_name):
    for output_layer_name in output_embedding_layer_name:

        output_embedding_layer = get_module(model, output_layer_name)
        if output_embedding_layer is None or not isinstance(output_embedding_layer, nn.Linear):
            logger.warning(f"Module {output_layer_name} does not exist")
            continue
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        module = CastOutputToFloat(output_embedding_layer)
        assert len(module) == 1
        set_module(model, output_layer_name, module[0])


def set_module(model, name, value):
    print(f"set module {name}")
    if "." in name and len(name.split(".")) == 2:
        module1, module2 = name.split(".")
        setattr(getattr(model, module1), module2, value)
        return
    setattr(model, name, value)

