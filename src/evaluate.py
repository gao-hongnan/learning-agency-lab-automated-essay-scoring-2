from rich.pretty import pprint
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="./data")

pprint(base_model)
