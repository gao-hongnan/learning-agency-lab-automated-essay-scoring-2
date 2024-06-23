from __future__ import annotations

import logging
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Config,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    StableDropout,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class DebertaV2WithAttentionPooler(DebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = AttentionPooler(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            pooler_hidden_dim_fc=config.hidden_size,
            pooler_dropout=config.pooler_dropout,
        )

        output_dim = self.pooler.output_dim
        output_dim = config.hidden_size
        self.classifier = nn.Linear(output_dim, num_labels)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.post_init()
        self.pooler.apply(init_attention_pooler)

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = True,
        return_dict: bool | None = None,
    ) -> Tuple[torch.FloatTensor, ...] | SequenceClassifierOutput:
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
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # FIXME: HF why even allow return hidden states if it is going to overflow gpu memory?
            attentions=None,
        )


# class DecoderBackbone(nn.Module):
#     backbone: PreTrainedModel | PeftModel

#     def __init__(
#         self,
#         composer: Composer,
#         *,
#         backbone: PreTrainedModel,
#         criterion: nn.Module | None = None,
#         eos_token_id: int | None = None,
#         tokenizer_size: int | None = None,
#     ) -> None:
#         super().__init__()
#         self.composer = composer
#         self.criterion = criterion
#         self.eos_token_id = eos_token_id
#         self.tokenizer_size = tokenizer_size

#         # get a backbone for our network
#         # Here let's go with AutoModel which will be AutoModelForCausalLM,
#         # it does not matter as we take
#         # remove the final layers
#         self.backbone = backbone

#         self.backbone.resize_token_embeddings(tokenizer_size)
#         # remove the head as we are going to use a custom head, this step is
#         # guardrail if one initialises from `AutoModelForCausalLM` and not from `AutoModel`
#         if hasattr(self.backbone, "lm_head"):
#             self.backbone.lm_head = nn.Identity()

#         if composer.shared.gradient_checkpointing:
#             self.enable_gradient_checkpointing()

#         if composer.shared.num_layers_to_remove is not None:
#             # we only remove the last layers as they can be superfluous: https://arxiv.org/html/2403.17887v1
#             try:
#                 getattr(self.backbone, "layers")
#             except AttributeError:
#                 raise ValueError(
#                     "The backbone model does not have the attribute `layers`. Print the model to see what is the correct attribute."
#                 )
#             self.backbone.layers = self.backbone.layers[
#                 : -composer.shared.num_layers_to_remove
#             ]

#         if self.composer.shared.freeze_embeddings:
#             logger.info("freezing embeddings.")
#             embedding_module = self.backbone.embed_tokens
#             self.freeze_layers(embedding_module)

#         if self.composer.shared.num_layers_to_freeze is not None and self.composer.shared.num_layers_to_freeze > 0:  # type: ignore[operator]
#             logger.info(
#                 "freezing the first %s layers.",
#                 self.composer.shared.num_layers_to_freeze,
#             )
#             # Here the first layers are frozen: only remaining last layers will be trained
#             for layer in self.backbone.layers[
#                 : self.composer.shared.num_layers_to_freeze
#             ]:
#                 self.freeze_layers(layer)

#         if self.composer.shared.reinitialize_n_layers > 0:
#             for module in self.backbone.layers[
#                 -self.composer.shared.reinitialize_n_layers :
#             ]:
#                 self._init_weights(module)

#         # if composer.shared.use_lora:
#         #     logger.info("Applying LoRA modifications to the backbone.")
#         #     self.backbone = self.apply_lora()
#         #     self.backbone.print_trainable_parameters()

#         # NOTE: put this after backbone init because in case of mutation
#         self.backbone_config = self.backbone.config

#         self.pool = ...

#         if self.composer.shared.task == "SINGLE_LABEL_CLASSIFICATION":
#             self.num_classes = composer.shared.num_classes
#             self.head = nn.Linear(
#                 self.backbone_config.hidden_size, composer.shared.num_classes
#             )

#         if self.composer.shared.task == "REGRESSION":
#             self.head = nn.Linear(self.backbone_config.hidden_size, 1)

#         if self.composer.shared.task == "CAUSAL_LM":
#             self.head = nn.Linear(
#                 self.backbone_config.hidden_size,
#                 self.backbone_config.vocab_size,
#             )

#     def _init_weights(self, module: nn.Module) -> None:
#         """You can add custom weight init logic here."""
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(
#                 mean=0.0, std=self.backbone_config.initializer_range
#             )
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(
#                 mean=0.0, std=self.backbone_config.initializer_range
#             )
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def enable_gradient_checkpointing(self) -> None:
#         """Enables gradient checkpointing for the backbone model."""
#         if self.backbone.supports_gradient_checkpointing:
#             logger.info("Enabling gradient checkpointing.")
#             self.backbone.gradient_checkpointing_enable()
#         else:
#             logger.warning(
#                 "Gradient checkpointing is not supported by the backbone model %s but you try to enable it.",
#                 self.backbone.__class__.__name__,
#             )

#     def init_backbone(self) -> None:
#         ...

#     def init_head(self) -> None:
#         ...

#     def freeze_layers(self, module: nn.Module) -> None:
#         """Freezes the specified number of layers in the backbone.

#         Args:
#             composer (object): Composer object with configuration details.
#         """
#         for parameter in module.parameters():
#             parameter.requires_grad = False

#     def apply_lora(self) -> nn.Module:
#         """Applies LoRA modifications to the backbone.

#         Args:
#             composer (object): Composer object with configuration details.
#         """
#         peft_config = LoraConfig(**self.composer.low_rank_config.model_dump())
#         self.backbone = get_peft_model(self.backbone, peft_config)
#         return self.backbone

#     def dry_run(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
#         """Dry run the model to check if the model is correctly set up."""
#         try:
#             with torch.no_grad():
#                 self.eval()
#                 self.forward(batch)
#                 self.train()
#             return {"status": "dry_run_ok"}

#         except Exception as exc:
#             logger.exception("Dry run failed with exception %s", str(exc))
#             return {"status": "dry_run_failed", "exception": str(exc)}


# class DecoderCausalLM(DecoderBackbone):
#     def forward(self, batch: dict[str, torch.Tensor]) -> CausalLMOutputWithPast:
#         pprint(batch)
#         # note carefuly
#         pprint(self.backbone)
#         backbone_outputs = self.backbone(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             return_dict=True,  # to force return `BaseModelOutputWithPast`
#         )[
#             "last_hidden_state"
#         ]  # see BaseModelOutputWithPast in MistralModel
#         pprint(backbone_outputs)

#         # feature = self.pool(...)
#         logits = self.head(backbone_outputs)
#         logits: torch.Tensor = logits.float()

#         loss = None
#         if "labels" in batch.keys():
#             # FIXME: follow hf style, but rly awkward to have shifting in model.
#             labels = batch["labels"]
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             shift_logits = shift_logits.view(-1, self.backbone_config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Ensure tensors are on the same device
#             shift_labels = shift_labels.to(shift_logits.device)
#             if self.criterion is None:
#                 self.criterion = nn.CrossEntropyLoss()

#             loss = self.criterion(shift_logits, shift_labels)

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=None,
#             hidden_states=None,
#             attentions=None,
#         )


# class DecoderClassifier(DecoderBackbone):
#     def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
#         x = batch["input_ids"]  # (bs, num_tokens)
#         # this assumes that you only have one eos_token per example
#         eos_positions = torch.argwhere(x == self.eos_token_id)[:, 1]
#         eos_count_per_example = torch.bincount(eos_positions[:, 0], minlength=x.size(0))
#         if torch.any(eos_count_per_example != 1):
#             raise ValueError(
#                 f"Expected exactly one EOS token per example for now got {eos_count_per_example}."
#             )

#         backbone_outputs = self.backbone(
#             input_ids=x,
#             attention_mask=batch["attention_mask"],
#         )

#         pprint(backbone_outputs.shape)

#         # feature = self.pool(...)

#         # we are only interested in the eos_token
#         backbone_outputs = backbone_outputs[
#             torch.arange(backbone_outputs.shape[0]), eos_positions
#         ]  # (bs, hidden_size)

#         logits = self.head(backbone_outputs)  # (bs, num_classes)

#         return {"logits": logits}
