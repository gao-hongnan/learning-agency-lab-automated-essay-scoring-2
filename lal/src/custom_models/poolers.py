from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Config, StableDropout


class ContextPooler(nn.Module):
    def __init__(self, config: DebertaV2Config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(
        self,
        backbone_outputs: BaseModelOutput,
        _input_ids: torch.Tensor | None = None,
        _attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        last_hidden_state: torch.Tensor = backbone_outputs.last_hidden_state
        context_token: torch.Tensor = last_hidden_state[:, 0]
        context_token: torch.Tensor = self.dropout(context_token)
        pooled_output: torch.Tensor = self.dense(context_token)
        pooled_output: torch.Tensor = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self) -> int:
        assert self.config.hidden_size and isinstance(self.config.hidden_size, int)
        return self.config.hidden_size


def init_attention_pooler(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)  # torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class AttentionPooler(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        pooler_hidden_dim_fc: int,
        pooler_dropout: float,
    ) -> None:
        """Initialize the AttentionPooler layer. Tested with `Deberta` model
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
        self.pooler_hidden_dim_fc = pooler_hidden_dim_fc
        self.pooler_dropout = pooler_dropout
        self.dropout = nn.Dropout(self.pooler_dropout)

        self.q = nn.Linear(self.hidden_size, 1, bias=False)  # weight.shape: (1, hidden_size)
        self.w_h = nn.Linear(
            self.hidden_size, self.pooler_hidden_dim_fc, bias=False
        )  # weight.shape: (pooler_hidden_dim_fc, hidden_size) input dim: hidden_size, output dim: pooler_hidden_dim_fc

        # nn.init.normal_(self.q_transform.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.w_h_transform.weight, mean=0.0, std=0.1)

    def forward(
        self,
        backbone_outputs: BaseModelOutput,
        _input_ids: torch.Tensor | None = None,
        _attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use deberta example:
        See `SequenceClassifierOutput` `hidden_states` shapeis a tuple of all
        layers hidden states - including the embedding layer - but we want the
        hidden layers.

         In `DebertaV2ForSequenceClassification`, we have:

         1. `outputs = self.deberta(...)` in `forward` method, call it `backbone_outputs`.
         2. `pooler` then takes in `backbone_outputs[0]` which is the last hidden state.
             We know it is the last hidden state by tracing to the return item of the
             backbone which is contained as a class `BaseModelOutput`:
             last_hidden_state (`torch.FloatTensor` of shape
             `(batch_size, sequence_length, hidden_size)`): Sequence of hidden-states at the output of the last layer of the model.
        3. So we know this shape is `(batch_size, sequence_length, hidden_size)`.
        """
        all_hidden_states = backbone_outputs.hidden_states
        # convert tuple of tensors to tensors
        all_hidden_states = torch.stack(all_hidden_states)  # type: ignore[assignment]
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
        out = self.attention(hidden_states, _attention_mask)
        out = self.dropout(out)
        return out

    def attention(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Notations.

        - B: batch size
        - L: number of hidden layers
        - D: hidden size
        - P: pooler_hidden_dim_fc
        - T: sequence length

        Note the `hidden_states` shape is `[B, L, D]` which is the CLS token
        hidden states for each layer. So we do not need to care about if the
        attention mask is present or not.

        他们这个pooling 就是 每个layer 和每个layer 搞下 看哪个layer softmax 最好
        """
        weights_q = self.q.weight  # [1, D]
        # hidden_states: [B, L, D] | hidden_states.transpose(-2, -1): [B, D, L]
        # weights_q @ hidden_states.transpose(-2, -1) -> [1, D] @ [B, D, L] -> broadcast -> [B, 1, D] @ [B, D, L] -> [B, 1, L]
        # v -> v.squeeze(1) -> [B, 1, L] -> [B, L] (this is each hidden layer's attention score)
        # we are treating hidden layers like a sequence of tokens and we are trying to get the attention score for each hidden layer
        v = torch.matmul(weights_q, hidden_states.transpose(-2, -1)).squeeze(1)

        # v: [B, L] -> F.softmax(v, dim=-1) -> [B, L] now softmaxed the attention scores to get the weights means which hidden layer to focus on
        v = F.softmax(v, dim=-1)

        # weights_w_h = self.w_h.weight  # [P, D]
        weights_w_h = self.w_h.weight

        # v.unsqueeze(1): [B, 1, L] | hidden_states: [B, L, D]
        # v.unsqueeze(1) @ hidden_states -> [B, 1, L] @ [B, L, D] -> [B, 1, D]
        # v_temp: [B, 1, D] -> v_temp.transpose(-2, -1) -> [B, D, 1]
        # weighted sum of hidden states based on the attention scores
        v_temp = torch.matmul(v.unsqueeze(1), hidden_states).transpose(-2, -1)
        # finally we are getting the context vector by multiplying the weighted sum of hidden states with the weights
        # weights_w_h @ v_temp -> [P, D] @ [B, D, 1] -> [B, P, 1] -> squeeze(2) -> [B, P]
        # means the final cls token representation
        v = torch.matmul(weights_w_h, v_temp).squeeze(2)
        return v

    # def attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
    #     # weights_q = self.q.weight # [1, hidden_size]
    #     # v = torch.matmul(weights_q, hidden_states.transpose(-2, -1)).squeeze(1)
    #     print("q shape", self.q.shape) # shape: (1, hidden_size)
    #     print("Initial hidden_states shape:", hidden_states.shape) # shape: (batch_size, num_hidden_layers, hidden_size)
    #     # Print the shape after transposing
    #     print("hidden_states transposed shape:", hidden_states.transpose(-2, -1).shape) # shape: (batch_size, hidden_size, num_hidden_layers)

    #     v = torch.matmul(self.q, hidden_states.transpose(-2, -1)).squeeze(1)

    #     print("Shape after applying q and squeezing:", v.shape) # shape: (batch_size, num_hidden_layers)
    #     v = F.softmax(v, dim=-1)
    #     print("Shape after softmax:", v.shape) # shape: (batch_size, num_hidden_layers)

    #     # weights_w_h = self.w_h.weight # shape: (hidden_size, pooler_hidden_dim_fc)
    #     print("Shape of v.unsqueeze(1):", v.unsqueeze(1).shape) # shape: (batch_size, 1, num_hidden_layers)
    #     print("Shape of hidden_states for context vector computation:", hidden_states.shape) # shape: (batch_size, num_hidden_layers, hidden_size)
    #     print("Resulting shape after matmul of v.unsqueeze(1) and hidden_states:", torch.matmul(v.unsqueeze(1), hidden_states).shape) # shape: (batch_size, 1, hidden_size)

    #     v_temp = torch.matmul(v.unsqueeze(1), hidden_states).transpose(-2, -1)
    #     print("Shape after transposing v_temp:", v_temp.shape) # shape: (batch_size, hidden_size, 1)
    #     print("Shape of w_h:", self.w_h.shape) # shape: (hidden_size, pooler_hidden_dim_fc)
    #     print("Shape of w_h transposed:", self.w_h.transpose(1, 0).shape) # shape: (pooler_hidden_dim_fc, hidden_size)
    #     print("Shape of matmul of w_h and v_temp:", torch.matmul(self.w_h.transpose(1, 0), v_temp).shape) # shape: (batch_size, pooler_hidden_dim_fc, 1)
    #     v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
    #     print("Final output shape after applying w_h:", v.shape)  # shape: (batch_size, pooler_hidden_dim_fc)
    #     return v

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


# class GemPooling(nn.Module):
#     def __init__(self, backbone_config, pooling_config):
#         super().__init__()

#         self.dim = backbone_config.hidden_size
#         self.eps = pooling_config.eps
#         self.p = Parameter(torch.ones(1) * pooling_config.p)

#         self.output_dim = backbone_config.hidden_size

#     def forward(self, inputs, backbone_output):
#         attention_mask = get_attention_mask(inputs)
#         x = get_last_hidden_state(backbone_output)

#         attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
#         x = torch.sum((x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p), 1)
#         ret = x / attention_mask_expanded.sum(1).clip(min=self.eps)
#         ret = ret.pow(1 / self.p)
#         return ret


# class MeanPooling(nn.Module):
#     def __init__(self, backbone_config, pooling_config):
#         super(MeanPooling, self).__init__()
#         self.output_dim = backbone_config.hidden_size

#     def forward(self, inputs, backbone_outputs):
#         attention_mask = get_attention_mask(inputs)
#         last_hidden_state = get_last_hidden_state(backbone_outputs)

#         input_mask_expanded = (
#             attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         )
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings
