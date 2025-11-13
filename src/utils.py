import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    data_path = Path(cfg["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print("Download CSV, columns:", df.columns.tolist())

    text_col = cfg["text_column"]
    label_col = cfg["label_column"]
    cols = df.columns.tolist()

    if text_col in cols and label_col in cols:
        df = df[[text_col, label_col]].dropna()
        return df

    combined_name = f"{text_col},{label_col}"
    if combined_name in cols:
        combined = df[combined_name].astype(str)

        split = combined.str.rsplit(",", n=1, expand=True)
        split.columns = [text_col, label_col]

        df = split.dropna()
        print("devide column", df.columns.tolist())
        return df

    raise KeyError(
        f"Expected columns '{text_col}' and '{label_col}' "
        f"or combined '{combined_name}' in CSV, found {cols}"
    )


def split_and_encode(df: pd.DataFrame, cfg):
    text_col = cfg["text_column"]
    label_col = cfg["label_column"]

    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_enc,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y_enc,
    )

    val_size_rel = cfg["val_size"] / (1 - cfg["test_size"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_size_rel,
        random_state=cfg["random_state"],
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, le
