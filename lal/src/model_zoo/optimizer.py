from typing import List, Type, Dict

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


def get_optimizer_grouped_parameters_by_category(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    layerwise_learning_rate_decay_mulitplier: float = 0.95,
    pooler_lr: float | None = None,
    head_lr: float | None = None,
    pooler_weight_decay: float | None = None,
    head_weight_decay: float | None = None,
) -> List[Dict[str, str | float | List[nn.Parameter]]]:
    # LayerNorm.bias is automatically included in no decay since bias is in no decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    embeddings_group = model.deberta.embeddings
    backbone_group = model.deberta.encoder.layer
    pooler_group = model.pooler
    head_group = model.classifier

    head_no_decay = {
        "params": [
            parameter
            for parameter_name, parameter in head_group.named_parameters()
            if any(nd in parameter_name for nd in no_decay)
        ],
        "weight_decay": 0.0,
        "lr": learning_rate if head_lr is None else head_lr,
        "name": "head_no_decay",
    }

    head_decay = {
        "params": [
            parameter
            for parameter_name, parameter in head_group.named_parameters()
            if not any(nd in parameter_name for nd in no_decay)
        ],
        "weight_decay": weight_decay if head_weight_decay is None else head_weight_decay,
        "lr": learning_rate if head_lr is None else head_lr,
        "name": "head_decay",
    }

    # this group applies no weight decay
    pooler_no_decay = {
        "params": [
            parameter
            for parameter_name, parameter in pooler_group.named_parameters()
            if any(nd in parameter_name for nd in no_decay)
        ],
        "weight_decay": 0.0,
        "lr": learning_rate if pooler_lr is None else pooler_lr,
        "name": "pooler_no_decay",
    }

    pooler_decay = {
        "params": [
            parameter
            for parameter_name, parameter in pooler_group.named_parameters()
            if not any(nd in parameter_name for nd in no_decay)
        ],
        "weight_decay": weight_decay if pooler_weight_decay is None else pooler_weight_decay,
        "lr": learning_rate if pooler_lr is None else pooler_lr,
        "name": "pooler_decay",
    }

    optimizer_grouped_parameters = [pooler_no_decay, pooler_decay, head_no_decay, head_decay]
    embeddings_and_backbone_group = [embeddings_group] + list(backbone_group)
    embeddings_and_backbone_group.reverse()

    lr = learning_rate
    for index, layer in enumerate(embeddings_and_backbone_group):
        lr *= layerwise_learning_rate_decay_mulitplier
        # NOTE: add no decay and decay groups for encoder/backbone

        optimizer_grouped_parameters += [
            {
                "params": [
                    parameter
                    for parameter_name, parameter in layer.named_parameters()
                    if not any(nd in parameter_name for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr,
                "name": f"{layer.__class__.__name__}_{index}_decay",
            },
            {
                "params": [
                    parameter
                    for parameter_name, parameter in layer.named_parameters()
                    if any(nd in parameter_name for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
                "name": f"{layer.__class__.__name__}_{index}_no_decay",
            },
        ]
    return optimizer_grouped_parameters


def get_optimizer_grouped_parameters_by_layer(
    model: nn.Module,
    group_configs: List[Dict[str, str | float | bool]],
    default_learning_rate: float,
    default_weight_decay: float,
    layerwise_learning_rate_decay_mulitplier: float = 0.95,
) -> List[Dict[str, str | float | List[nn.Parameter]]]:
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameter_groups = []
    named_parameters = list(model.named_parameters())

    for parameter_name, parameter in named_parameters:
        weight_decay = 0.0 if any(nd in parameter_name for nd in no_decay) else default_weight_decay
        applied = False
        for group_config in group_configs:
            if parameter_name.startswith(group_config["prefix"]):
                print(f"Applying {group_config['prefix']} to {parameter_name}")
                layer_lr = group_config.get("base_lr", default_learning_rate)

                if group_config.get("llrd", False):
                    layer_lr *= layerwise_learning_rate_decay_mulitplier
                    optimizer_parameter_groups.append(
                        {
                            "params": parameter,
                            "weight_decay": weight_decay,
                            "lr": layer_lr,
                            "name": f"{group_config['prefix']}_decay",
                        }
                    )
                else:
                    optimizer_parameter_groups.append(
                        {
                            "params": parameter,
                            "weight_decay": weight_decay,
                            "lr": layer_lr,
                            "name": f"{group_config['prefix']}_decay",
                        }
                    )
                applied = True
                break

        if not applied:
            optimizer_parameter_groups.append(
                {
                    "params": parameter,
                    "weight_decay": weight_decay,
                    "lr": default_learning_rate,
                    "name": "default",
                }
            )

    return optimizer_parameter_groups
