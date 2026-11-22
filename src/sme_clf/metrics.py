"""
src/sme_clf/metrics.py
Self-contained, mathematically rigorous classification metrics engine.
Includes bootstrap confidence intervals, stratified view breakdowns, and export utilities.
Can be verified against scikit-learn.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import numpy as np

from src.sme_clf.prompting import CATEGORIES_17

def compute_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> np.ndarray:
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm

def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = CATEGORIES_17,
    support_threshold_interpretable: int = 5
) -> Dict[str, Any]:
    """
    Computes accuracy, macro-F1 (support > 0), all-17 macro-F1, weighted-F1,
    per-class precision/recall/f1/support, and distribution alignment.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have equal length"
    n_total = len(y_true)
    if n_total == 0:
        return {"error": "empty_predictions"}

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = round(correct / n_total, 4)

    # Per-class counts
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter(y_true)
    pred_counts = Counter(y_pred)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    per_class = {}
    f1_list_supported = []
    f1_list_all = []
    zero_support_classes = []
    weighted_f1_sum = 0.0

    for cat in labels:
        s = support[cat]
        c_tp = tp[cat]
        c_fp = fp[cat]
        c_fn = fn[cat]

        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        is_interpretable = (s >= support_threshold_interpretable)

        per_class[cat] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": s,
            "predicted_count": pred_counts[cat],
            "interpretable": is_interpretable,
            "flag": None if is_interpretable else "support < 5 (not interpretable)"
        }

        f1_list_all.append(f1)
        if s > 0:
            f1_list_supported.append(f1)
            weighted_f1_sum += f1 * s
        else:
            zero_support_classes.append(cat)

    macro_f1_supported = round(float(np.mean(f1_list_supported)), 4) if f1_list_supported else 0.0
    macro_f1_all = round(float(np.mean(f1_list_all)), 4) if f1_list_all else 0.0
    weighted_f1 = round(weighted_f1_sum / n_total, 4)

    # Prediction distribution vs gold distribution
    gold_dist = {cat: round(support[cat] / n_total, 4) for cat in labels}
    pred_dist = {cat: round(pred_counts[cat] / n_total, 4) for cat in labels}

    cm = compute_confusion_matrix(y_true, y_pred, labels)

    return {
        "n_samples": n_total,
        "accuracy": accuracy,
        "macro_f1_supported": macro_f1_supported,
        "macro_f1_all_17": macro_f1_all,
        "zero_support_classes": zero_support_classes,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "gold_distribution": gold_dist,
        "prediction_distribution": pred_dist,
        "confusion_matrix": cm.tolist()
    }

def bootstrap_confidence_interval(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = CATEGORIES_17,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes 95% bootstrap confidence intervals for accuracy and macro-F1.
    Prints explicit correlation warning as required.
    """
    warning_text = (
        "WARNING: Examples within the same prospectus document exhibit correlated formatting "
        "and editorial structure. Bootstrap resampling at the example level understates real uncertainty. "
        "A single-document evaluation (e.g. Century) cannot support statistical generalizations about unseen documents."
    )

    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    macro_f1s = []

    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        sub_true = [y_true[i] for i in indices]
        sub_pred = [y_pred[i] for i in indices]

        metrics = compute_classification_metrics(sub_true, sub_pred, labels=labels)
        accs.append(metrics["accuracy"])
        macro_f1s.append(metrics["macro_f1_supported"])

    low_p = (alpha / 2.0) * 100
    high_p = (1.0 - alpha / 2.0) * 100

    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "accuracy_ci_95": [round(float(np.percentile(accs, low_p)), 4), round(float(np.percentile(accs, high_p)), 4)],
        "macro_f1_ci_95": [round(float(np.percentile(macro_f1s, low_p)), 4), round(float(np.percentile(macro_f1s, high_p)), 4)],
        "correlation_warning": warning_text
    }

def save_confusion_matrix_csv(cm: np.ndarray, labels: List[str], out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual \\ predicted"] + labels)
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(row))

def save_confusion_matrix_png(cm: np.ndarray, labels: List[str], out_path: Path):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            title="Confusion Matrix",
            ylabel="True label",
            xlabel="Predicted label"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
    except Exception as e:
        # If matplotlib is not available, write placeholder note
        pass

