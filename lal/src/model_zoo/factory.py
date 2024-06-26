from __future__ import annotations

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Config

from .criterion import OrdinalRegressionLoss, RegLossForClassification
from .pooling import AttentionPooler, ContextPooler, GemPooler, MeanPooler

import torch

def get_pooler(config: DebertaV2Config) -> nn.Module:
    """Factory method to get pooler based on the config."""
    if not hasattr(config, "pooler_type") or config.pooler_type == "context" or config.pooler_type is None:
        return ContextPooler(config)

    elif config.pooler_type == "attention":
        return AttentionPooler(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            pooler_hidden_dim_fc=getattr(config, "pooler_hidden_dim_fc", config.hidden_size),
            pooler_dropout=config.pooler_dropout,
        )

    elif config.pooler_type == "mean":
        return MeanPooler(output_dim=config.hidden_size)
    elif config.pooler_type == "gem":
        pooler_config = getattr(config, "pooler_config", {})
        return GemPooler(
            p=pooler_config.get("p", 3),
            eps=pooler_config.get("eps", 1e-6),
            output_dim=config.hidden_size,
        )
    else:
        raise ValueError(f"Pooler {config.pooler_type} is not supported.")


def get_loss(config: DebertaV2Config) -> nn.Module:
    """Factory method to get loss based on the config."""
    if config.criterion is None:
        if config.problem_type == "regression":
            return MSELoss(**config.criterion_config)
        if config.problem_type == "single_label_classification":
            if config.criterion_config.get("weight", None) is not None:
                # NOTE: to solve json not serializable issue we pop the weight.
                weight = torch.as_tensor(config.criterion_config.pop("weight"))
            return CrossEntropyLoss(weight=weight, **config.criterion_config)
        if config.problem_type == "multi_label_classification":
            return BCEWithLogitsLoss(**config.criterion_config)

    if config.criterion == "mse":
        return MSELoss(**config.criterion_config)
    if config.criterion == "cross_entropy":
        if config.criterion_config.get("weight", None) is not None:
            # NOTE: to solve json not serializable issue we pop the weight.
            weight = torch.as_tensor(config.criterion_config.pop("weight"))
        return CrossEntropyLoss(**config.criterion_config)
    if config.criterion == "bce":
        return BCEWithLogitsLoss(**config.criterion_config)
    if config.criterion == "reg_cls_loss":
        return RegLossForClassification(**config.criterion_config)
    if config.criterion == "huber":
        # see intuition: https://www.kaggle.com/code/emiz6413/cv-0-825-lb-0-803-deberta-v3-small-with-huber-loss
        return nn.HuberLoss(**config.criterion_config)
    if config.criterion == "ordinal_reg_loss":
        return OrdinalRegressionLoss(**config.criterion_config)
    raise ValueError(f"Criterion {config.criterion} is not supported.")
