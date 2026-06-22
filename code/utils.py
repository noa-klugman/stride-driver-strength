from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold


RANDOM_STATE = 42
MODEL_DIR = Path("../model")
OUTPUT_DIR = Path("../output_data")
FIGURE_DIR = Path("../figures")

TVA_COLUMN = "TVA"
GROUP_COLUMNS = ["genename", "aapos"]

# Columns that identify the variant or are not used as model features.
NON_FEATURE_COLUMNS = [
    "Entry",
    "aapos",
    "aaalt",
    "genename",
    "aaref",
    "AAC_1L",
    "TVA"
]


def make_directories():
    for directory in [MODEL_DIR, OUTPUT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def build_xy(drivers: pd.DataFrame, tva_threshold: float):
    """Create feature matrix and weak/strong labels for a given TVA threshold."""
    y = (drivers[TVA_COLUMN] >= tva_threshold).astype(int)
    X = drivers.drop(columns=NON_FEATURE_COLUMNS)
    return X, y


def split_train_validation_test(X: pd.DataFrame, y: pd.Series):
    """
    Split data using grouped strategy.
    Groups are defined by gene and amino-acid position, ensuring mutations from the same protein position are kept in the same split.
    First split: training vs temporary set using 3 folds: 66.7% training.
    Second split: temporary set into validation and test using 2 folds: 16.7% validation and 16.7% test.
    """
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_idx, tmp_idx = next(splitter.split(X, y, groups=X["group"]))

    X_train, X_tmp = X.iloc[train_idx], X.iloc[tmp_idx]
    y_train, y_tmp = y.iloc[train_idx], y.iloc[tmp_idx]

    splitter = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    val_idx, test_idx = next(splitter.split(X_tmp, y_tmp, groups=X_tmp["group"]))

    X_test, X_val = X_tmp.iloc[test_idx], X_tmp.iloc[val_idx]
    y_test, y_val = y_tmp.iloc[test_idx], y_tmp.iloc[val_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_threshold_scan(results_df: pd.DataFrame):
    """Save a simple plot summarizing validation balanced accuracy across TVA thresholds."""
    best_idx = results_df["validation_balanced_accuracy"].idxmax()
    best_row = results_df.loc[best_idx]

    plt.figure(figsize=(7, 5))
    plt.plot(
        results_df["tva_threshold"],
        results_df["validation_balanced_accuracy"],
        marker="o",
        linestyle="--",
    )
    plt.scatter(
        [best_row["tva_threshold"]],
        [best_row["validation_balanced_accuracy"]],
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )
    plt.xlabel("TVA threshold")
    plt.ylabel("Validation balanced accuracy")
    plt.title("TVA threshold scan")
    plt.xticks(results_df["tva_threshold"])
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tva_threshold_scan.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_prob = model.predict_proba(X_test.drop(columns=["group"], errors="ignore"))[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_value = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auc_value:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("STRIDE ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "stride_roc_auc.png", dpi=300, bbox_inches="tight")
    plt.close()
