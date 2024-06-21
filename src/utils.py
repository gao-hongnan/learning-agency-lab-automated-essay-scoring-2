from __future__ import annotations

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

from src.logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)


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

    if task in ["CLASSIFICATION", "REGRESSION"]:
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


def dry_run(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Dry run the model to check if the model is correctly set up."""
    try:
        with torch.no_grad():
            self.eval()
            self.forward(batch)
            self.train()
        return {"status": "dry_run_ok"}

    except Exception as exc:
        logger.exception("Dry run failed with exception %s", str(exc))
        return {"status": "dry_run_failed", "exception": str(exc)}
