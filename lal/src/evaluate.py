from rich.pretty import pprint
from transformers import AutoModelForSequenceClassification


base_model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path="/Users/gaohn/gaohn/learning-agency-lab-automated-essay-scoring-2/artifacts/output_v20240621175434"
)
# base_model = DebertaWithAttentionPooling.from_pretrained(
#     pretrained_model_name_or_path="/Users/gaohn/gaohn/learning-agency-lab-automated-essay-scoring-2/artifacts/output_v20240621183258"
# )

pprint(base_model)
