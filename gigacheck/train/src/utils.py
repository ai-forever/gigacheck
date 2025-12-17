from loguru import logger


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable(%): {100 * trainable_params / all_param}"
    )


def save_config(config, pretrained_model_name, id2label, output_dir):
    config.pretrained_model_name_or_path = pretrained_model_name
    config.id2label = id2label
    config.save_pretrained(output_dir)
    logger.info(f"Model config saved to {output_dir}")
