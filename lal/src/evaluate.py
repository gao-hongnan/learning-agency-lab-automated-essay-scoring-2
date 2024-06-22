import torch
from rich.pretty import pprint
from transformers import AutoModelForSequenceClassification

from .models import AttentionPooler, DebertaV2WithAttentionPooler

base_model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path="/Users/gaohn/omniverse/learning-agency-lab-automated-essay-scoring-2/artifacts/output_v20240621230426/checkpoint-64"
)

base_model.pooler = AttentionPooler(
    num_hidden_layers=base_model.config.num_hidden_layers,
    hidden_size=base_model.config.hidden_size,
    pooler_hidden_dim_fc=base_model.config.hidden_size,
    pooler_dropout=base_model.config.hidden_dropout_prob,
)

score_weights = torch.load(
    "/Users/gaohn/omniverse/learning-agency-lab-automated-essay-scoring-2/artifacts/output_v20240621230426/checkpoint-64/base_model_with_pooler.pt",
    map_location="cpu",
)
base_model.load_state_dict(score_weights)

# from .src.models import AttentionPooler, DebertaV2WithAttentionPooler


# base_model = DebertaWithAttentionPooling.from_pretrained(
#     pretrained_model_name_or_path="/Users/gaohn/gaohn/learning-agency-lab-automated-essay-scoring-2/artifacts/output_v20240621183258"
# )

pprint(base_model)
