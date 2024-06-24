from typing import Any, Dict

import numpy as np
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score
from transformers import EvalPrediction


def compute_metrics_for_classification(eval_pred: EvalPrediction) -> Dict[str, Any]:
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    eval_qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    results = {"eval_qwk": eval_qwk}
    return results


def compute_metrics_for_regression(eval_pred: EvalPrediction) -> Dict[str, Any]:
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    eval_qwk = cohen_kappa_score(labels, predictions.clip(0, 5).round(0), weights="quadratic")
    results = {"eval_qwk": eval_qwk}
    return results


def compute_metrics_for_reg_cls(eval_pred: EvalPrediction) -> Dict[str, Any]:
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    bs = labels.shape[0]

    probs = softmax(predictions, axis=-1)
    indices = np.stack([np.arange(0, 6) for _ in range(bs)])
    scores = (probs * indices).sum(axis=-1)

    eval_qwk = cohen_kappa_score(labels, scores.clip(0, 5).round(0), weights="quadratic")
    results = {"eval_qwk": eval_qwk}
    return results
