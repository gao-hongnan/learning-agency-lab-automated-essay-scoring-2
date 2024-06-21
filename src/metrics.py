from typing import Any, Dict

from sklearn.metrics import cohen_kappa_score
from transformers import EvalPrediction


def compute_metrics_for_classification(eval_pred: EvalPrediction) -> Dict[str, Any]:
    predictions, labels = eval_pred
    eval_qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    results = {"eval_qwk": eval_qwk}
    return results


def compute_metrics_for_regression(eval_pred: EvalPrediction) -> Dict[str, Any]:
    predictions, labels = eval_pred
    eval_qwk = cohen_kappa_score(labels, predictions.clip(0, 5).round(0), weights="quadratic")
    results = {"eval_qwk": eval_qwk}
    return results
