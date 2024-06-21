from __future__ import annotations

import logging
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def get_all_hidden_states(backbone_outputs):
    all_hidden_states = torch.stack(backbone_outputs[1])
    return all_hidden_states


class AttentionPooling(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        pooler_hidden_dim_fc: int,
        pooler_dropout: float = 0.0,
    ):
        """Initialize the AttentionPooling layer. Tested with `Deberta` model
        but not limited to it. You may have to change some config names if you
        want to use it for decoder only models like `Mistral`.

        Parameters
        ----------
        num_hidden_layers : int
            The number of hidden layers in the transformer model. Usually can
            be found in the base/backbone model config `config.num_hidden_layers`.
        hidden_size : int
            The hidden size of the transformer model. Usually can be found in
            the base/backbone model config `config.hidden_size`.
        pooler_hidden_dim_fc : int
            The hidden dimension of the fully connected layer after the
            attention pooling. This is the output dimension of the pooling
            layer. Needed to we can decide
            the shape of the head layer. We can find this usually from the
            base/backbone model config `config.hidden_size`. If you want to
            overhaul the head, you can change this value but make sure shape
            matches.
        pooler_dropout : float, optional
            The dropout rate for the pooling layer, by default 0.0. Set this to
            same as `config.pooler_dropout` usually.

        References
        ----------
        [1] https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
        """
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.pooler_dropout = pooler_dropout

        self.pooler_hidden_dim_fc = pooler_hidden_dim_fc
        self.dropout = nn.Dropout(self.pooler_dropout)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.pooler_hidden_dim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

    def forward(self, backbone_outputs):
        """Use deberta example:
         In `DebertaV2ForSequenceClassification`, we have:

         1. `outputs = self.deberta(...)` in `forward` method, call it `backbone_outputs`.
         2. `pooler` then takes in `backbone_outputs[0]` which is the last hidden state.
             We know it is the last hidden state by tracing to the return item of the
             backbone which is contained as a class `BaseModelOutput`:
             last_hidden_state (`torch.FloatTensor` of shape
             `(batch_size, sequence_length, hidden_size)`): Sequence of hidden-states at the output of the last layer of the model.
        3. So we know this shape is `(batch_size, sequence_length, hidden_size)`.
        """
        self.q = self.q.to(backbone_outputs[0].device)
        self.w_h = self.w_h.to(backbone_outputs[0].device)

        # all layers hidden states
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack(
            [
                all_hidden_states[layer_i][:, 0].squeeze()
                for layer_i in range(
                    1, self.num_hidden_layers + 1
                )  # why from range 1 is because first layer is the embed layer
            ],
            dim=-1,
        )
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

    @property
    def output_dim(self) -> int:
        """You typically need to add this property because models like
        `Deberta` expects the output dimension of the pooling layer. See
        example below where they call `self.pooler.output_dim`.

        ```python
        class DebertaForSequenceClassification(DebertaPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)

                num_labels = getattr(config, "num_labels", 2)
                self.num_labels = num_labels

                self.deberta = DebertaModel(config)
                self.pooler = ContextPooler(config)
                output_dim = self.pooler.output_dim

                self.classifier = nn.Linear(output_dim, num_labels)
        ```
        """

        return self.pooler_hidden_dim_fc


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

#         if self.composer.shared.task == "CLASSIFICATION":
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
