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

import torch
from torch import nn, optim
from typing import Dict, Iterator, Tuple, Set


def check_optimizer_coverage(model: nn.Module, optimizer: optim.Optimizer) -> None:
    """
    Checks if all parameters in the given model are covered by the optimizer's parameter groups.

    Args:
        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer linked to the model.

    Prints:
        Outputs the names of parameters that are not covered by any optimizer group.
        If all parameters are covered, it prints a confirmation message.
    """
    # Gather all model parameters with names
    model_params: Dict[str, nn.Parameter] = {name: param for name, param in model.named_parameters()}

    # Gather all optimizer parameters
    opt_params: Set[nn.Parameter] = set(param for group in optimizer.param_groups for param in group["params"])

    # Check if all parameters are covered
    uncovered_params: Dict[str, nn.Parameter] = {
        name: param for name, param in model_params.items() if param not in opt_params
    }

    if uncovered_params:
        print("Some parameters are not covered by the optimizer:")
        for name in uncovered_params:
            print(name)
    else:
        print("All parameters are covered by the optimizer.")


def get_optimizer_grouped_parameters_by_category(
    model: nn.Module,
    base_learning_rate: float,
    default_weight_decay: float,
    layerwise_learning_rate_decay_mulitplier: float = 0.95,
    pooler_lr: float | None = None,
    head_lr: float | None = None,
    pooler_weight_decay: float | None = None,
    head_weight_decay: float | None = None,
) -> List[Dict[str, str | float | List[nn.Parameter]]]:
    # LayerNorm.bias is automatically included in no decay since bias is in no decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameter_groups = []

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
        "lr": base_learning_rate if head_lr is None else head_lr,
        "name": "head_no_decay",
    }

    head_decay = {
        "params": [
            parameter
            for parameter_name, parameter in head_group.named_parameters()
            if not any(nd in parameter_name for nd in no_decay)
        ],
        "weight_decay": default_weight_decay if head_weight_decay is None else head_weight_decay,
        "lr": base_learning_rate if head_lr is None else head_lr,
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
        "lr": base_learning_rate if pooler_lr is None else pooler_lr,
        "name": "pooler_no_decay",
    }

    pooler_decay = {
        "params": [
            parameter
            for parameter_name, parameter in pooler_group.named_parameters()
            if not any(nd in parameter_name for nd in no_decay)
        ],
        "weight_decay": default_weight_decay if pooler_weight_decay is None else pooler_weight_decay,
        "lr": base_learning_rate if pooler_lr is None else pooler_lr,
        "name": "pooler_decay",
    }

    optimizer_parameter_groups = [pooler_no_decay, pooler_decay, head_no_decay, head_decay]
    embeddings_and_backbone_group = [embeddings_group] + list(backbone_group)
    embeddings_and_backbone_group.reverse()

    embeddings_and_backbone_learning_rate = base_learning_rate
    for index, layer in enumerate(embeddings_and_backbone_group):
        embeddings_and_backbone_learning_rate *= layerwise_learning_rate_decay_mulitplier
        # NOTE: add no decay and decay groups for encoder/backbone

        optimizer_parameter_groups += [
            {
                "params": [
                    parameter
                    for parameter_name, parameter in layer.named_parameters()
                    if not any(nd in parameter_name for nd in no_decay)
                ],
                "weight_decay": default_weight_decay,
                "lr": embeddings_and_backbone_learning_rate,
                "name": f"{layer.__class__.__name__}_{index}_decay",
            },
            {
                "params": [
                    parameter
                    for parameter_name, parameter in layer.named_parameters()
                    if any(nd in parameter_name for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": embeddings_and_backbone_learning_rate,
                "name": f"{layer.__class__.__name__}_{index}_no_decay",
            },
        ]
    return optimizer_parameter_groups
