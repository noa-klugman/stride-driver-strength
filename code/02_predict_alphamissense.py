from pathlib import Path

import joblib
import pandas as pd


INPUT_PATH = Path("../raw_data/alphamissense_pathogenic_data.csv")
MODEL_PATH = Path("../model/stride_random_forest.pkl")
MODEL_SUMMARY_PATH = Path("../output_data/final_model_summary.csv")
OUTPUT_PATH = Path("../output_data/alphamissense_stride_predictions.csv")

BINARY_COLUMNS = ["Bind_ligand", "PTM", "structured", "in_pocket"]

NON_FEATURE_COLUMNS = [
    "Entry",
    "aapos",
    "aaalt",
    "genename",
    "aaref",
    "AAC_1L"
]


def load_prediction_table(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)

    df[BINARY_COLUMNS] = df[BINARY_COLUMNS].astype(int)
    df["RSA"] = pd.to_numeric(df["RSA"], errors="coerce")

    return df


def build_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        col for col in NON_FEATURE_COLUMNS
        if col in df.columns
    ]

    X = df.drop(columns=columns_to_drop)

    return X


def main():
    model = joblib.load(MODEL_PATH)

    model_summary = pd.read_csv(MODEL_SUMMARY_PATH)
    decision_threshold = float(model_summary.loc[0, "decision_threshold"])

    variants = load_prediction_table(INPUT_PATH)
    # Build model input features by removing identifier/non-feature columns.
    X = variants.drop(columns=NON_FEATURE_COLUMNS)

    predicted_probability = model.predict_proba(X)[:, 1]
    predicted_label = (predicted_probability >= decision_threshold).astype(int)

    predictions = variants[["genename", "AAC_1L"]].copy()

    predictions["predicted_strength_group"] = pd.Series(
        predicted_label,
        index=variants.index
    ).map({
        0: "Weak",
        1: "Strong",
    })

    predictions.to_csv(OUTPUT_PATH, index=False)

    print(predictions["predicted_strength_group"].value_counts())


if __name__ == "__main__":
    main()
