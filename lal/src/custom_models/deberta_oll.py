from transformers.models.deberta_v2.modeling_deberta_v2 import *
from torch import nn
import numpy as np
import torch.nn.functional as F

class DebertaV2OLL(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels
        self.beta = 1.5

        self._init_dist_matrix()

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)


        # Initialize weights and apply final processing
        self.post_init()

    def _init_dist_matrix(self):
        x = torch.arange(self.num_labels)
        matrix = torch.abs(x.view(-1, 1) - x.view(1, -1)).to(self.config.torch_dtype)

        self.dist_matrix = nn.Parameter(matrix, requires_grad=False)

        # self.dist_matrix.requires_grad = False
        
        
    def _compute_oll_loss(self):
        pass

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        
        if labels is not None:
            num_classes = self.num_labels
            dist_matrix = self.dist_matrix

            probas = F.softmax(logits, dim=1)
            true_labels = torch.stack([labels] * num_classes, dim=1)
            label_ids = torch.stack([torch.arange(num_classes)] * len(labels))
            distances = dist_matrix[true_labels, label_ids].float()
            distances_tensor = distances.clone().requires_grad_(True).to(probas.device)

            err = - torch.log(1 - probas) * distances_tensor ** (self.beta)
            loss = torch.sum(err, axis=1).mean()
            
    
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )