import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from .utils import load_config, load_data, split_and_encode


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    return total_loss / len(data_loader), correct / total


def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = correct / total if total > 0 else 0.0
    return total_loss / len(data_loader), acc, np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_data(cfg)

    if args.limit is not None and args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)), random_state=cfg["random_state"])

    X_train, X_val, X_test, y_train, y_val, y_test, le = split_and_encode(df, cfg)

    model_name = cfg.get("bert_model_name", "distilbert-base-uncased")
    max_len = cfg.get("max_len", 128)
    batch_size = cfg.get("batch_size", 16)
    lr = cfg.get("lr", 2e-5)
    epochs = args.epochs

    print(f"Using transformer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = TextDataset(X_train, y_train, tokenizer, max_len)
    val_ds = TextDataset(X_val, y_val, tokenizer, max_len)
    test_ds = TextDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(le.classes_),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, device)

        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    # фінальна оцінка на train / val / test
    _, train_acc, train_labels, train_preds = eval_epoch(model, train_loader, device)
    _, val_acc, val_labels, val_preds = eval_epoch(model, val_loader, device)
    _, test_acc, test_labels, test_preds = eval_epoch(model, test_loader, device)

    print("\nAccuracy:")
    print(f"  train: {train_acc:.4f}")
    print(f"  val  : {val_acc:.4f}")
    print(f"  test : {test_acc:.4f}\n")

    print("Classification report (test):")
    print(classification_report(test_labels, test_preds, target_names=le.classes_))

    cm = confusion_matrix(test_labels, test_preds)
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
        "history": history,
        "model_name": model_name,
        "max_len": max_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    with open(results_dir / "bert_transformer.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    save_dir = results_dir / "distilbert_model"
    save_dir.mkdir(exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"\nResults and model saved in: {save_dir}")


if __name__ == "__main__":
    main()
