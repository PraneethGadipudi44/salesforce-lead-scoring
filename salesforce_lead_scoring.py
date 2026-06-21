from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_URL = "https://raw.githubusercontent.com/wcrowley342/LogisticRegression/main/Lead%20Source%20Dataset%20V14.csv"
DATA_PATH = DATA_DIR / "lead_source_dataset_v14.csv"

NUMERIC_FEATURES = ["WebsiteVisits", "CompanySize", "EmailOpens", "EmailClicks"]
CATEGORICAL_FEATURES = ["State", "LeadSource"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    ensure_directories()
    if not DATA_PATH.exists():
        urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


def load_and_prepare_data() -> pd.DataFrame:
    dataset_path = download_dataset()
    df = pd.read_csv(dataset_path)

    df["Customer"] = df["Customer"].astype(str).str.strip()
    df["is_converted"] = df["Customer"].map({"Yes": 1, "No": 0})

    df["State"] = df["State"].fillna("Unknown").astype(str).str.strip()
    df["LeadSource"] = df["LeadSource"].fillna("Unknown").astype(str).str.strip()

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    prepared = df[MODEL_FEATURES + ["is_converted"]].dropna(subset=["is_converted"]).copy()
    return prepared


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def evaluate_models(df: pd.DataFrame) -> dict[str, object]:
    X = df[MODEL_FEATURES]
    y = df["is_converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    metrics_rows: list[dict[str, float | str]] = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "auc_roc": roc_auc_score(y_test, probabilities),
            }
        )
        trained_models[model_name] = pipeline

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["auc_roc", "f1_score", "precision"],
        ascending=False,
    )
    best_model_name = metrics_df.iloc[0]["model"]

    return {
        "X_test": X_test,
        "y_test": y_test,
        "metrics_df": metrics_df,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
    }


def recommend_action(conversion_probability: float) -> str:
    if conversion_probability >= 0.75:
        return "High Priority"
    if conversion_probability >= 0.35:
        return "Medium Priority"
    return "Low Priority"


def score_lead(model: Pipeline, lead_attributes: dict[str, object]) -> dict[str, object]:
    lead_frame = pd.DataFrame([lead_attributes], columns=MODEL_FEATURES)
    probability = float(model.predict_proba(lead_frame)[0, 1])
    lead_score = round(probability * 100, 1)

    return {
        "lead_attributes": lead_attributes,
        "conversion_probability": round(probability, 4),
        "lead_score": lead_score,
        "recommended_action": recommend_action(probability),
    }


def sample_leads() -> list[dict[str, object]]:
    return [
        {
            "State": "California",
            "LeadSource": "Adwords",
            "WebsiteVisits": 20,
            "CompanySize": 70,
            "EmailOpens": 18,
            "EmailClicks": 9,
        },
        {
            "State": "Texas",
            "LeadSource": "Print",
            "WebsiteVisits": 8,
            "CompanySize": 70,
            "EmailOpens": 10,
            "EmailClicks": 4,
        },
        {
            "State": "Ohio",
            "LeadSource": "Tradeshow",
            "WebsiteVisits": 4,
            "CompanySize": 180,
            "EmailOpens": 2,
            "EmailClicks": 0,
        },
    ]


def save_outputs(
    prepared_df: pd.DataFrame,
    evaluation_results: dict[str, object],
    sample_results: list[dict[str, object]],
) -> None:
    prepared_df.to_csv(OUTPUT_DIR / "prepared_leads.csv", index=False)
    evaluation_results["metrics_df"].to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    with (OUTPUT_DIR / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            evaluation_results["metrics_df"].to_dict(orient="records"),
            handle,
            indent=2,
        )

    with (OUTPUT_DIR / "sample_scoring_output.txt").open("w", encoding="utf-8") as handle:
        handle.write("Sample Lead Scoring Output\n")
        handle.write("=" * 27 + "\n\n")
        for index, result in enumerate(sample_results, start=1):
            handle.write(f"Lead {index}\n")
            handle.write(f"Attributes: {result['lead_attributes']}\n")
            handle.write(
                "Result: "
                f"probability={result['conversion_probability']}, "
                f"lead_score={result['lead_score']}, "
                f"action={result['recommended_action']}\n\n"
            )


def main() -> None:
    prepared_df = load_and_prepare_data()
    evaluation_results = evaluate_models(prepared_df)
    best_model = evaluation_results["best_model"]
    best_model_name = evaluation_results["best_model_name"]
    metrics_df = evaluation_results["metrics_df"]

    sample_results = [score_lead(best_model, lead) for lead in sample_leads()]
    save_outputs(prepared_df, evaluation_results, sample_results)

    print("Salesforce Lead Scoring Prototype")
    print("=" * 34)
    print(f"Dataset rows used: {len(prepared_df):,}")
    print(f"Public source: {DATA_URL}")
    print(
        "Dataset note: this public lead file includes lead source, company size, "
        "state, and engagement behavior, but it does not include an explicit industry field."
    )
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nChampion model selected for scoring: {best_model_name}")

    print("\nSample scoring demonstration:")
    for index, result in enumerate(sample_results, start=1):
        print(f"Lead {index}:")
        print(f"  Attributes: {result['lead_attributes']}")
        print(
            "  Output: "
            f"probability={result['conversion_probability']}, "
            f"lead_score={result['lead_score']}, "
            f"action={result['recommended_action']}"
        )


if __name__ == "__main__":
    main()
