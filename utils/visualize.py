"""
utils/visualize.py — Графики обучения, confusion matrix, feature importance
"""
import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from config import LOGS_DIR, OUTPUTS_DIR, IDX_TO_LABEL


def plot_training_history(history_path: Path = LOGS_DIR / "training_history.json"):
    with open(history_path) as f:
        h = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    metrics = [
        ("loss",  "Loss",     "train_loss", "val_loss"),
        ("acc",   "Accuracy", "train_acc",  "val_acc"),
        ("f1",    "Macro F1", "train_f1",   "val_f1"),
    ]
    for ax, (_, title, tr_key, val_key) in zip(axes, metrics):
        ax.plot(h[tr_key],  label="Train", linewidth=2)
        ax.plot(h[val_key], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUTS_DIR / "training_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Сохранён: {path}")


def plot_confusion_matrix(cm_path: Path = LOGS_DIR / "confusion_matrix.npy",
                          normalize: bool = True):
    cm = np.load(cm_path)
    n  = cm.shape[0]
    labels = [IDX_TO_LABEL.get(i, str(i)).split("___")[-1].replace("_", " ")
              for i in range(n)]

    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", None

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=vmax, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""),
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Сохранён: {path}")


def plot_rf_feature_importance(rf_model):
    fi = rf_model.feature_importance()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
    bars    = ax.barh(fi["feature"], fi["importance"], color=colors)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, val in zip(bars, fi["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()

    path = OUTPUTS_DIR / "rf_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Сохранён: {path}")


def plot_risk_heatmap(csv_path: Path):
    """Тепловая карта средней вероятности болезни по регион×сезон."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    pivot = df.groupby(["region", "season"])["disease_probability"].mean().unstack()
    season_order = ["Весна", "Лето", "Осень", "Зима"]
    pivot = pivot.reindex(columns=[c for c in season_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=0.7, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Avg Disease Probability"})
    ax.set_title("Средняя вероятность болезни: Регион × Сезон",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = OUTPUTS_DIR / "risk_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Сохранён: {path}")
