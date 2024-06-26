from __future__ import annotations

import logging
from typing import Any, Tuple

import torch
from omnivault.utils.torch_utils.model_utils import get_named_modules
from rich.pretty import pprint
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Config,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    StableDropout,
)

from ..logger import get_logger
from .factory import get_loss, get_pooler
from omnivault.utils.torch_utils.model_utils import Freezer

logger = get_logger(__name__, level=logging.DEBUG)


def _init_weights(module: nn.Module, **kwargs: Any) -> None:
    """Initialize the weights. This is a helper function to initialize the weights
    of a torch model.

    Parameters
    ----------
    module : nn.Module
        The module to initialize the weights. It can be the model itself, or
        a sub-module of the model such as nn.Linear, nn.Embedding, nn.LayerNorm.
    kwargs : Any
        Additional keyword arguments to pass to the initialization method.
        This is like a _universal_ way to accept configuration.
    """
    # std = self.config.initializer_range
    init_method = kwargs.get("init_method", "normal")

    if isinstance(module, nn.Linear):
        if init_method == "normal":
            module.weight.data.normal_(mean=kwargs.get("mean", 0.0), std=kwargs.get("std", 0.02))
        elif init_method == "xavier_uniform":
            module.weight.data = nn.init.xavier_uniform_(module.weight.data, gain=kwargs.get("gain", 1.0))
        elif init_method == "xavier_normal":
            module.weight.data = nn.init.xavier_normal_(module.weight.data, gain=kwargs.get("gain", 1.0))
        elif init_method == "kaiming_uniform":
            module.weight.data = nn.init.kaiming_uniform_(
                module.weight.data,
                kwargs.get("a", 0),
                kwargs.get("mode", "fan_in"),
                kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif init_method == "kaiming_normal":
            module.weight.data = nn.init.kaiming_normal_(
                module.weight.data,
                kwargs.get("a", 0),
                kwargs.get("mode", "fan_in"),
                kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif init_method == "orthogonal":
            module.weight.data = nn.init.orthogonal_(module.weight.data, kwargs.get("gain", 1.0))

        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        if init_method == "normal":
            module.weight.data.normal_(mean=kwargs.get("mean", 0.0), std=kwargs.get("std", 0.02))
        elif init_method == "xavier_uniform":
            module.weight.data = nn.init.xavier_uniform_(module.weight.data, gain=kwargs.get("gain", 1.0))
        elif init_method == "xavier_normal":
            module.weight.data = nn.init.xavier_normal_(module.weight.data, gain=kwargs.get("gain", 1.0))
        elif init_method == "kaiming_uniform":
            module.weight.data = nn.init.kaiming_uniform_(
                module.weight.data,
                kwargs.get("a", 0),
                kwargs.get("mode", "fan_in"),
                kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif init_method == "kaiming_normal":
            module.weight.data = nn.init.kaiming_normal_(
                module.weight.data,
                kwargs.get("a", 0),
                kwargs.get("mode", "fan_in"),
                kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif init_method == "orthogonal":
            module.weight.data = nn.init.orthogonal_(module.weight.data, kwargs.get("gain", 1.0))

        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class SubclassedDebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    """We can overload deberta's config with `criterion` and `pooler_type` along
    maybe with `loss_config` and `pooler_config` to customize the loss and pooler
    for the sequence classification task.


    Notes
    -----
    [1] `if self.config.reinitialize_n_layers_of_backbone > 0: ...`
        We are reinit weights of backbone's encoder so only works for deberta/bert
        with an encoder layer named `layer` attribute. If want to use on other
        models, just change the attribute name. But what are we doing here?
        The reason is we load from pre-trained, and sometimes we may want to
        reset/recalibrate the weights of the backbone (i.e. last N BLOCK)
        for stable training. Note this is not reinit pooler or classifier since
        they will be reinitialised anyways! So if we want init last 3 BLOCK of layers
        then we can set `reinitialize_n_layers_of_backbone=3` in config.
    """

    def __init__(self, config: DebertaV2Config) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        # 1. LOAD BACKBONE
        self.deberta = DebertaV2Model(config)  # NOTE: alias=self.backbone
        self.freezer = Freezer(self.deberta)

        if self.config.enable_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing.")
            self.deberta.gradient_checkpointing_enable()

        if self.config.freeze_embeddings:
            logger.info("freezing embeddings.")
            embedding_module = self.deberta.embeddings
            self.freezer.freeze_by_module(embedding_module)

        if self.config.freeze_these_layers_indices: # [1, 2]
            logger.info("freezing the specified layers %s.", self.config.freeze_these_layers_indices)
            self.freezer.freeze_by_index(self.config.freeze_these_layers_indices, "encoder.layer")

        pprint(self.freezer.report_freezing())

        if self.config.reinitialize_n_layers_of_backbone > 0:
            for module in self.deberta.encoder.layer[-self.config.reinitialize_n_layers_of_backbone :]:
                logger.info("Reinitializing weights of %s", module.__class__.__name__)
                self._init_weights(module)

        # 2. LOAD POOLER
        self.pooler = get_pooler(config)  # Factory method to get pooler
        output_dim = self.pooler.output_dim

        # 3. LOAD REGRESSOR/CLASSIFIER HEAD
        self.classifier = nn.Linear(output_dim, num_labels)  # NOTE: alias=self.head

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights. This overrides base class."""
        init_config = getattr(self.config, "init_config", {})
        _init_weights(module, **init_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.deberta.get_input_embeddings()  # type: ignore[no-any-return]

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.deberta.set_input_embeddings(new_embeddings)

    def _get_loss(self) -> nn.Module:
        return get_loss(self.config)  # Factory method to get loss

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> Tuple[torch.Tensor, ...] | SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        backbone_outputs: BaseModelOutput = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.pooler(
            backbone_outputs=backbone_outputs, _input_ids=input_ids, _attention_mask=attention_mask
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = self._get_loss()
            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + backbone_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # NOTE: assume we do not need computational graph for hidden states and attentions - as it will
        # cause major OOM during training/evaluation.
        hidden_states = None
        attentions = None
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
