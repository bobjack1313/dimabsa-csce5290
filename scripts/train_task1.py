#!/usr/bin/env python3
# =============================================================================
# Script Name : train_task1.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
# Description : Train a BERT-based regression model for DimASR (Task 1).
#
# This script:
#   • Loads processed data from data/processed/task1/
#   • Extracts VA_gold → [V, A] regression labels
#   • Tokenizes text using a HuggingFace model
#   • Trains a 2-output regression head (valence, arousal)
#   • Saves the final model to experiments/checkpoints/task1/bert_final/
#
# Authors     : Amrit Adhikari, Bob Jack - Group 4
# Date        : 2025-11-12
# =============================================================================
import argparse
from pathlib import Path
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ---- CLI Setup ----
def parse_args():
    parser = argparse.ArgumentParser(
        description = "Train DimABSA Task 1 (DimASR) model"
    )
    p.add_argument("--model", type=str, default="bert-base-uncased")

    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/checkpoints/task1"),
        help="Save location for Task 1 checkpoints",
    )

    return parser.parse_args()


# ---- Loading the Dataset ----
def load_task1():
    base = Path("data/processed/task1")
    splits = {}

    for split in ["train", "valid"]:
        path = base / f"{split}.jsonl"
        if not path.exists():
            continue

        rows = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                example = json.loads(line)

                # VA_gold → labels
                gold = example.get("VA_gold", {})
                v, a = gold.get("V"), gold.get("A")
                if v is None or a is None:
                    example["labels"] = torch.tensor([5.0, 5.0])
                else:
                    example["labels"] = torch.tensor([v, a])

                rows.append(example)

        splits[split] = Dataset.from_list(rows)

    if not splits:
        raise RuntimeError("No task1 dataset found under data/processed/task1/")

    return DatasetDict(splits)


# ----Tokenizing ----
def build_preprocess(tokenizer):

    def func(batch):
        enc = tokenizer(
            batch["Text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        return enc

    return func



def main():
    args = parse_args()

    print("[Task1] Loading dataset...")
    dataset = load_task1()

    print("[Task1] Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    preprocess = build_preprocess(tokenizer)
    dataset = dataset.map(preprocess, batched=True)

    # Configure regression head (2 outputs: V and A)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        problem_type="regression",
    )

    # Training Args
    train_args = TrainingArguments(
        output_dir=str(args.out_dir),
        eval_strategy="epoch",
        do_eval=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("valid", None),
        tokenizer=tokenizer,
    )

    print("[Task1] Training...")
    trainer.train()

    save_path = args.out_dir / "bert_final"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"[Task1] Saving final model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("[Task1] Done.")


# Main entry point
if __name__ == "__main__":
    main()
