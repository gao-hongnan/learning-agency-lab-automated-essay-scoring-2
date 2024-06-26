from typing import Any, Dict

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score
from transformers import EvalPrediction


def logits_to_classes(logits: torch.Tensor, cutpoints: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to ordinal classes based on cutpoints.

    Parameters:
    - logits: Model output logits (shape: [batch_size])
    - cutpoints: Cutpoints tensor used for ordinal regression (shape: [num_classes - 1])

    Returns:
    - Ordinal class predictions (shape: [batch_size])
    """
    # Calculate cumulative logits
    cumulative_logits = logits.unsqueeze(1) - cutpoints.unsqueeze(0)

    # Apply sigmoid to get probabilities
    probas = torch.sigmoid(cumulative_logits)

    # Sum probabilities to get predicted class (assumes cutpoints define class boundaries)
    class_predictions = probas.sum(dim=1).round().clip(0, len(cutpoints))

    return class_predictions


def compute_metrics_for_classification(eval_pred: EvalPrediction) -> Dict[str, Any]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    eval_qwk = cohen_kappa_score(labels, logits.argmax(-1), weights="quadratic")
    results = {"eval_qwk": eval_qwk}
    return results


def compute_metrics_for_regression(eval_pred: EvalPrediction) -> Dict[str, Any]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    eval_qwk = cohen_kappa_score(labels, logits.clip(0, 5).round(0), weights="quadratic")
    results = {"eval_qwk": eval_qwk}
    return results


def compute_metrics_for_ordinal_regression(eval_pred: EvalPrediction, cutpoints: torch.Tensor) -> Dict[str, Any]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = logits_to_classes(torch.tensor(logits), cutpoints)
    eval_qwk = cohen_kappa_score(labels, predictions, weights="quadratic")
    results = {
        "eval_qwk": eval_qwk,
    }

    return results


def compute_metrics_for_reg_cls(eval_pred: EvalPrediction) -> Dict[str, Any]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    bs = labels.shape[0]

    probs = softmax(logits, axis=-1)
    indices = np.stack([np.arange(0, 6) for _ in range(bs)])
    scores = (probs * indices).sum(axis=-1)

    eval_qwk = cohen_kappa_score(labels, scores.clip(0, 5).round(0), weights="quadratic")
    results = {"eval_qwk": eval_qwk}
    return results
