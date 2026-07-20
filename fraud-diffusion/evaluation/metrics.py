import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Geometric mean of sensitivity (recall on class 1) and specificity (recall on class 0)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(sensitivity * specificity))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "g_mean": g_mean(y_true, y_pred),
    }
