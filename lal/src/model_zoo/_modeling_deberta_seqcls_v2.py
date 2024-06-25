from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Config,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    StableDropout,
)

from .factory import get_loss, get_pooler


class SubclassedDebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    """We can overload deberta's config with `criterion` and `pooler_type` along
    maybe with `loss_config` and `pooler_config` to customize the loss and pooler
    for the sequence classification task.
    """

    def __init__(self, config: DebertaV2Config) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)

        self.pooler = get_pooler(config)  # Factory method to get pooler
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

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
