from __future__ import annotations

import json
import logging
from typing import Any

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from .logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)


def jsonify(obj: Any) -> Any:
    """Converts the object to a JSON serializable format.

    Parameters
    ----------
    obj : Any
        Object to convert to JSON serializable format.

    Returns
    -------
    Any
        JSON serializable object.
    """

    return json.dumps(obj, indent=4)


def load_tokenizer(
    pretrained_model_name_or_path: str, **kwargs: Any
) -> PreTrainedTokenizerBase | PreTrainedTokenizerFast:
    """Load tokenizer from huggingface.

    Parameters
    ----------
    pretrained_model_name_or_path : str
        Model name to load from huggingface.
    kwargs : Any
        Additional keyword arguments to pass to the tokenizer.

    Returns
    -------
    PreTrainedTokenizerBase
        Tokenizer loaded from huggingface.
    """
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)


def load_model(
    pretrained_model_name_or_path: str,
    load_backbone_only: bool = False,
    task: str = "CAUSAL_LM",
    **kwargs: Any,
) -> PreTrainedModel:
    """Load model from huggingface.

    Parameters
    ----------
    pretrained_model_name_or_path : str
        Model name to load from huggingface.
    kwargs : Any
        Additional keyword arguments to pass to the model.

    Returns
    -------
    PreTrainedModel
        Decoder for Causal Modeling loaded from huggingface.

    Note
    ----
    This factory always return an instance of `AutoModelForCausalLM` class and
    this is not restrictive because we can perform model surgery as needed if
    we want to say, change the model's architecture to become a classifier. All
    we need to do is change the `lm_head` of this backbone model to a classifier
    head.
    """

    if load_backbone_only and not task:
        logger.warning("Ignoring `task` as `load_backbone_only` is set to True.")

    if load_backbone_only:  # NOTE: this early returns anyways regardless of task
        return AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )

    if task in ["SINGLE_LABEL_CLASSIFICATION", "REGRESSION"]:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )

    if task == "CAUSAL_LM":
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )

    if task == "QA":
        return AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )


def dry_run(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Dry run the model to check if the model is correctly set up."""
    try:
        with torch.no_grad():
            model.eval()
            outputs = model.forward(**batch)
            model.train()
        return {"status": "SUCCESS", "outputs": outputs}

    except Exception as exc:
        logger.exception("Dry run failed with exception %s", str(exc))
        return {"status": "FAILED", "exception": str(exc)}


def get_parameters_groups(n_layers, n_groups):
    layers = [f"backbone.encoder.layer.{n_layers - i - 1}." for i in range(n_layers)]
    step = math.ceil(n_layers / n_groups)
    groups = []
    for i in range(0, n_layers, step):
        if i + step >= n_layers - 1:
            group = layers[i:]
            groups.append(group)
            break
        else:
            group = layers[i : i + step]
            groups.append(group)
    return groups


def get_grouped_llrd_parameters(model, encoder_lr, decoder_lr, embeddings_lr, lr_mult_factor, weight_decay, n_groups):
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    n_layers = model.backbone_config.num_hidden_layers
    parameters_groups = get_parameters_groups(n_layers, n_groups)

    for _, (name, params) in enumerate(named_parameters):
        wd = 0.0 if any(p in name for p in no_decay) else weight_decay

        if name.startswith("backbone.encoder"):
            lr = encoder_lr
            for i, group in enumerate(parameters_groups):
                lr = encoder_lr * (lr_mult_factor ** (i + 1)) if any(p in name for p in group) else lr

            opt_parameters.append({"params": params, "weight_decay": wd, "lr": lr})

        if name.startswith("backbone.embeddings"):
            lr = embeddings_lr
            opt_parameters.append({"params": params, "weight_decay": wd, "lr": lr})

        if name.startswith("bigram_type_embeddings"):
            lr = embeddings_lr
            opt_parameters.append({"params": params, "weight_decay": wd, "lr": lr})

        if (
            name.startswith("fc")
            or name.startswith("backbone.pooler")
            or name.startswith("pool")
            or name.startswith("pooling")
        ):
            lr = decoder_lr
            opt_parameters.append({"params": params, "weight_decay": wd, "lr": lr})

    return opt_parameters


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if ("backbone" not in n) and ("backbone_prompt" not in n)
            ],
            "lr": decoder_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


def get_optimizer(model, config):
    if config.optimizer.group_lr_multiplier == 1:
        optimizer_parameters = get_optimizer_params(
            model,
            config.optimizer.encoder_lr,
            config.optimizer.decoder_lr,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        optimizer_parameters = get_grouped_llrd_parameters(
            model,
            encoder_lr=config.optimizer.encoder_lr,
            decoder_lr=config.optimizer.decoder_lr,
            embeddings_lr=config.optimizer.embeddings_lr,
            lr_mult_factor=config.optimizer.group_lr_multiplier,
            weight_decay=config.optimizer.weight_decay,
            n_groups=config.optimizer.n_groups,
        )

    optimizer = AdamW(
        optimizer_parameters,
        lr=config.optimizer.encoder_lr,
        eps=config.optimizer.eps,
        betas=[config.optimizer.beta1, config.optimizer.beta2],
    )
    return optimizer


