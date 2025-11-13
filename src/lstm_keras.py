import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .utils import load_config, load_data, split_and_encode


def build_model(vocab_size: int, max_len: int, num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(128)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_data(cfg)

    if args.limit is not None and args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)), random_state=cfg["random_state"])

    X_train, X_val, X_test, y_train, y_val, y_test, le = split_and_encode(df, cfg)

    max_features = cfg.get("max_features", 20000)
    max_len = cfg.get("max_len", 128)

    vectorizer = TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )

    print("Adaptation on train-data")
    vectorizer.adapt(tf.constant(X_train))

    def vectorize(texts):
        return vectorizer(tf.constant(list(texts))).numpy()

    Xtr = vectorize(X_train)
    Xva = vectorize(X_val)
    Xte = vectorize(X_test)

    num_classes = len(le.classes_)

    model = build_model(
        vocab_size=max_features,
        max_len=max_len,
        num_classes=num_classes,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    print("Training BiLSTM model")
    history = model.fit(
        Xtr,
        np.array(y_train),
        validation_data=(Xva, np.array(y_val)),
        epochs=cfg.get("epochs", 5),
        batch_size=cfg.get("batch_size", 32),
        callbacks=callbacks,
        verbose=2,
    )
    print("Training completed\n")

    # predictions
    def predict_classes(x):
        probs = model.predict(x, verbose=0)
        return np.argmax(probs, axis=1)

    y_train_pred = predict_classes(Xtr)
    y_val_pred = predict_classes(Xva)
    y_test_pred = predict_classes(Xte)

    # accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("Accuracy:")
    print(f"  train: {train_acc:.4f}")
    print(f"  val  : {val_acc:.4f}")
    print(f"  test : {test_acc:.4f}\n")

    # classification report (test)
    print("Classification report (test):")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    # confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix (test):")
    print("classes:", le.classes_.tolist())
    print(cm)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results = {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "classes": le.classes_.tolist(),
        "confusion_matrix": cm.tolist(),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }

    with open(results_dir / "lstm_keras.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    model.save(results_dir / "lstm_model.h5")

    print(f"\nResults and model saved in folder: {results_dir}")


if __name__ == "__main__":
    main()
