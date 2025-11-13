import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from .utils import load_config, load_data, split_and_encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_data(cfg)

    if args.limit is not None:
        df = df.sample(n=min(args.limit, len(df)), random_state=cfg["random_state"])

    X_train, X_val, X_test, y_train, y_val, y_test, le = split_and_encode(df, cfg)

    # Модель: TF-IDF + Logistic Regression
    # verbose=1 дає стандартні логи від sklearn під час навчання
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=cfg["max_features"])),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            verbose=1,
        )),
    ])

    print(" Training model TF-IDF + LogisticRegression...")
    pipe.fit(X_train, y_train)
    print("Training completed\n")

    # Оцінка на train / val / test
    y_train_pred = pipe.predict(X_train)
    y_val_pred = pipe.predict(X_val)
    y_test_pred = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("Accuracy):")
    print(f"  train: {train_acc:.4f}")
    print(f"  val  : {val_acc:.4f}")
    print(f"  test : {test_acc:.4f}\n")

    # Детальний звіт по тесту
    print("Classification report (test):")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    # Матриця сплутування
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix (test):")
    print("classes:", le.classes_.tolist())
    print(cm)

    # результати в JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results = {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "classes": le.classes_.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    with open(results_dir / "baseline_sklearn.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'baseline_sklearn.json'}")


if __name__ == "__main__":
    main()
