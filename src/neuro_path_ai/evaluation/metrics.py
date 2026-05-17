from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    sensitivity: float
    specificity: float
    auc_roc: float | None
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


def compute_binary_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
) -> BinaryMetrics:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)

    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())

    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    auc_roc: float | None = None
    if y_score is not None:
        y_score_arr = np.asarray(list(y_score), dtype=float)
        if len(np.unique(y_true_arr)) == 2:
            auc_roc = float(roc_auc_score(y_true_arr, y_score_arr))

    return BinaryMetrics(
        sensitivity=sensitivity,
        specificity=specificity,
        auc_roc=auc_roc,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )
