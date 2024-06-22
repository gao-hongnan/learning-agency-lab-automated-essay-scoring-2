from transformers.models.deberta_v2.modeling_deberta_v2 import *
from torch import nn
import numpy as np
import torch.nn.functional as F

class AttentionPooler(nn.Module):
    def __init__(self, config, hiddendim_fc):
        super(AttentionPooler, self).__init__()

        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.hiddendim_fc = hiddendim_fc

        self.q = nn.Linear(self.hidden_size, 1, bias=False)
        self.w_h = nn.Linear(self.hidden_size, self.hiddendim_fc, bias=False)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

    def attention(self, h):
        v = self.q(h).squeeze()
        v = F.softmax(v, -1)

        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.weight, v_temp).squeeze(2)
        
        return v

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        
        out = self.attention(hidden_states)
        out = self.dropout(out)

        return out

        
class DebertaWithAttentionPooling(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)


        self.deberta = DebertaV2Model(config)
        self.pooler = AttentionPooler(config, hiddendim_fc=self.config.hidden_size)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels
        
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        
        

        
        # drop_out = getattr(config, "cls_dropout", None)
        # drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        # self.dropout = StableDropout(drop_out)

        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        all_hidden_states = outputs[1]
        pooled_output = self.pooler(all_hidden_states)
        # pooled_output = self.dropout(pooled_output)

        bs = pooled_output.shape[0]

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
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
