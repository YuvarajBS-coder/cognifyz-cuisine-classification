"""Train a cuisine classification model from Dataset.csv."""
from pathlib import Path
import json
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Dataset.csv"
MODEL_PATH = ROOT / "cuisine_model.joblib"

CATEGORICAL = ["City", "Currency", "Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
NUMERIC = ["Country Code", "Average Cost for two", "Price range", "Aggregate rating", "Votes"]
FEATURES = CATEGORICAL + NUMERIC


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="replace")
    df = df.dropna(subset=["Cuisines"]).copy()
    # The dataset can list several cuisines. Use the first listed cuisine as the target label.
    df["Target Cuisine"] = df["Cuisines"].astype(str).str.split(",").str[0].str.strip()
    df = df[df["Target Cuisine"].ne("")]
    # Keep the 15 most frequent labels to make the multiclass model practical and reduce extreme sparsity.
    top_labels = df["Target Cuisine"].value_counts().head(15).index.tolist()
    df = df[df["Target Cuisine"].isin(top_labels)].copy()
    X = df[FEATURES].copy()
    y = df["Target Cuisine"]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, NUMERIC), ("categorical", categorical_pipe, CATEGORICAL)])
    model = Pipeline([("preprocessor", prep), ("classifier", RandomForestClassifier(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    joblib.dump(model, MODEL_PATH)

    # Save a compact report and confusion-matrix image for the project report/GitHub.
    (ROOT / "metrics.json").write_text(json.dumps({
        "target_definition": "First cuisine listed in the Cuisines column",
        "classes_used": top_labels,
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "classification_report": report,
    }, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=top_labels, xticks_rotation=90, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Cuisine Classification — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(ROOT / "graphs" / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"Rows used: {len(df)}")
    print(f"Classes: {len(top_labels)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {ROOT / 'metrics.json'}")

if __name__ == "__main__":
    main()
