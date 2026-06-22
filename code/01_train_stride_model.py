"""
Train the STRIDE driver-strength classifier.

1. Load feature table for TVA-supported drivers.
2. Scan TVA thresholds used to define weak/strong drivers.
3. For each TVA threshold, tune a Random Forest by cross-validation on the training set.
4. Tune the decision threshold on the training set.
5. Select the TVA threshold with the best validation balanced accuracy.
6. Retrain the final model and evaluate it on the held-out test set.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.model_selection import TunedThresholdClassifierCV
from utils import (
    RANDOM_STATE,
    MODEL_DIR,
    OUTPUT_DIR,
    FIGURE_DIR,
    make_directories,
    build_xy,
    split_train_validation_test,
    plot_threshold_scan,
    plot_roc_curve
)


INPUT_PATH = Path("../raw_data/stride_training_variants_features.csv")

BINARY_COLUMNS = ["Bind_ligand", "PTM", "structured", "in_pocket"]

RF_PARAM_DISTRIBUTION = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 20),
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 6),
}


def load_training_table(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df[BINARY_COLUMNS] = df[BINARY_COLUMNS].astype(int)
    df["RSA"] = pd.to_numeric(df["RSA"], errors="coerce")
    df["group"] = df["genename"].astype(str) + "_" + df["aapos"].astype(str)
    return df


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    """Tune Random Forest hyperparameters using grouped cross-validation and AUROC."""
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=RF_PARAM_DISTRIBUTION,
        n_iter=20,
        scoring="roc_auc",
        cv=cv.split(X_train, y_train, groups=X_train["group"]),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train.drop(columns=["group"]), y_train)

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_decision_threshold(model, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Tune the probability cutoff to maximize balanced accuracy."""
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    tuned = TunedThresholdClassifierCV(
        model,
        scoring="balanced_accuracy",
        cv=cv.split(X_train, y_train, groups=X_train["group"]),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    tuned.fit(X_train.drop(columns=["group"]), y_train)

    return float(tuned.best_threshold_)


def evaluate_model(model, threshold: float, X: pd.DataFrame, y: pd.Series, set_name: str):
    """Evaluate a fitted classifier using the tuned probability threshold."""
    X_features = X.drop(columns=["group"], errors="ignore")
    y_prob = model.predict_proba(X_features)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "set": set_name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "auc": roc_auc_score(y, y_prob),
    }

    print(f"\n{set_name} set")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"AUC: {metrics['auc']:.3f}")

    return metrics, y_prob, y_pred


def scan_tva_thresholds(drivers: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Run the full model-selection workflow for each candidate TVA threshold."""
    results = []

    for tva_threshold in thresholds:
        print("\n" + "=" * 70)
        print(f"TVA threshold: {tva_threshold:.1f}")

        X, y = build_xy(drivers, tva_threshold)
        print("Label counts:")
        print(y.value_counts().rename(index={0: "Weak", 1: "Strong"}))

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(X, y)
        model, best_params, cv_auc = tune_random_forest(X_train, y_train)
        decision_threshold = tune_decision_threshold(model, X_train, y_train)

        print(f"Best CV AUROC: {cv_auc:.3f}")
        print(f"Tuned decision threshold: {decision_threshold:.3f}")
        print(f"Best parameters: {best_params}")

        train_metrics, _, _ = evaluate_model(model, decision_threshold, X_train, y_train, "Training")
        val_metrics, _, _ = evaluate_model(model, decision_threshold, X_val, y_val, "Validation")

        results.append({
            "tva_threshold": round(float(tva_threshold), 1),
            "n_weak": int((y == 0).sum()),
            "n_strong": int((y == 1).sum()),
            "cv_auc": cv_auc,
            "decision_threshold": decision_threshold,
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_auc": train_metrics["auc"],
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_auc": val_metrics["auc"],
            "best_params": best_params,
        })

    results_df = pd.DataFrame(results)
    return results_df


def train_final_model(drivers: pd.DataFrame, selected_tva_threshold: float):
    """Retrain the final model after threshold selection and evaluate on the held-out test set."""
    X, y = build_xy(drivers, selected_tva_threshold)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(X, y)

    X_final_train = pd.concat([X_train, X_val], axis=0)
    y_final_train = pd.concat([y_train, y_val], axis=0)

    final_model, best_params, cv_auc = tune_random_forest(X_final_train, y_final_train)
    decision_threshold = tune_decision_threshold(final_model, X_final_train, y_final_train)

    test_metrics, y_prob, y_pred = evaluate_model(
        final_model,
        decision_threshold,
        X_test,
        y_test,
        "Held-out test",
    )

    test_predictions = X_test.copy()
    test_predictions["true_label"] = y_test.values
    test_predictions["predicted_probability_strong"] = y_prob
    test_predictions["predicted_label"] = y_pred

    joblib.dump(final_model, MODEL_DIR / "stride_random_forest.pkl")

    metadata = pd.DataFrame([{
        "selected_tva_threshold": selected_tva_threshold,
        "decision_threshold": decision_threshold,
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_auc": test_metrics["auc"],
        "best_params": best_params,
    }])
    metadata.to_csv(OUTPUT_DIR / "final_model_summary.csv", index=False)

    plot_roc_curve(final_model, X_test, y_test)

    return final_model


def main():
    make_directories()

    drivers = load_training_table(INPUT_PATH)

    thresholds = np.round(np.arange(2.0, 3.01, 0.1), 1)
    results_df = scan_tva_thresholds(drivers, thresholds)
    plot_threshold_scan(results_df)

    best_row = results_df.loc[results_df["validation_balanced_accuracy"].idxmax()]
    selected_tva_threshold = float(best_row["tva_threshold"])

    print("\n" + "=" * 70)
    print("Selected TVA threshold")
    print(f"TVA threshold: {selected_tva_threshold:.1f}")
    print(f"Validation balanced accuracy: {best_row['validation_balanced_accuracy']:.3f}")

    train_final_model(drivers, selected_tva_threshold)


if __name__ == "__main__":
    main()
