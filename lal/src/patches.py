# 4.41.2 transformers patches
from typing import Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput


def deberta_v2_seq_cls_forward(
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
) -> Tuple | SequenceClassifierOutput:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs: BaseModelOutput = self.deberta(
        input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    all_hidden_states: Tuple[torch.FloatTensor, ...] = outputs.hidden_states

    pooled_output = self.pooler(all_hidden_states)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                # regression task
                loss_fn = nn.MSELoss()
                logits = logits.view(-1).to(labels.dtype)
                loss = loss_fn(logits, labels.view(-1))
            elif labels.dim() == 1 or labels.size(-1) == 1:
                label_index = (labels >= 0).nonzero()
                labels = labels.long()
                if label_index.size(0) > 0:
                    labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
                    labels = torch.gather(labels, 0, label_index.view(-1))
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                else:
                    loss = torch.tensor(0).to(logits)
            else:
                log_softmax = nn.LogSoftmax(-1)
                loss = -((log_softmax(logits) * labels).sum(-1)).mean()
        elif self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    # FIXME: HF why even allow return hidden states if it is going to overflow gpu memory?
    outputs.hidden_states = None
    outputs.attentions = None
    return SequenceClassifierOutput(
        loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
    )
