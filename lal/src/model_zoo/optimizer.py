from typing import List, Type

from torch import nn

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
BLACKLISTED: List[str] = ["bias", "LayerNorm.weight", "LayerNorm.bias"]  # CHANGE AS YOU WISH


def get_parameter_names(model: nn.Module, forbidden_layer_types: List[Type[nn.Module]]) -> List[str]:
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model: nn.Module) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


### USAGE ###
# decay_parameters = get_decay_parameter_names(BASE_MODEL)
# optimizer_grouped_parameters = [
#     {"params": [p for n, p in BASE_MODEL.named_parameters() if (n in decay_parameters and p.requires_grad)]},
#     {"params": [p for n, p in BASE_MODEL.named_parameters() if n not in decay_parameters and p.requires_grad]},
# ]
# optimizer = AdamW(
#     optimizer_grouped_parameters,
#     lr=1e-5,
#     weight_decay=0.01,
#     betas=(0.9, 0.999),
#     eps=1e-6,
#     capturable=False,
#     differentiable=False,
#     maximize=False,
#     amsgrad=False,
# )

# LLRD


def get_optimizer_grouped_parameters(model: nn.Module, learning_rate, weight_decay, layerwise_learning_rate_decay):
    # LayerNorm.bias is automatically included in no decay since bias is in no decay
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    embeddings = model.deberta.embeddings
    backbone = model.deberta.encoder.layer

    layers = [embeddings] + list(backbone)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters
