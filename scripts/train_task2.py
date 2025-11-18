#!/usr/bin/env python3
# =============================================================================
# Script Name : train_task2.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
# Description : Train a seq2seq baseline for Track A, Subtask 2 (DimASTE).
#
# Task 2 (DimASTE):
#   Given a review sentence, extract all (Aspect, Opinion, VA) triplets.
#
# This script:
#   • Loads processed data from data/processed/task2/{train,valid}.jsonl
#   • Builds text targets from the Quadruplet field.
#   • Trains a T5-style seq2seq model to generate those targets.
#   • Saves the final model in experiments/checkpoints/task2/t5_final/
#
# Authors     : Amrit Adhikari, Bob Jack - Group 4
# Date        : 2025-11-12
# =============================================================================

import argparse
from pathlib import Path
import json

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)


# ---- CLI Setup ----
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train seq2seq baseline for DimABSA Task 2 (DimASTE)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="t5-small",
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
        default=Path("experiments/checkpoints/task2"),
        help="Save location for Task 2 checkpoints",
    )

    return parser.parse_args()


# ---- Loading the Dataset ----
def load_task2() -> DatasetDict:
    '''
    Load processed Task 2 data from data/processed/task2/{train,valid}.jsonl.

    Expected schema (from prepare_datasets.py):

        {
          "ID": str,
          "Text": str,
          "Quadruplet": [
              {
                "Aspect": str,
                "Category": str,
                "Opinion": str,
                "VA": "V#A"
              },
              ...
          ]
        }

    Returns:
        DatasetDict with 'train' and 'valid' splits.
    '''
    base = Path("data/processed/task2")
    splits = {}

    for split in ["train", "valid"]:
        path = base / f"{split}.jsonl"
        if not path.exists():
            continue

        rows = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                example = json.loads(line)
                rows.append(example)

        splits[split] = Dataset.from_list(rows)

    if not splits:
        raise RuntimeError("No Task 2 data found under data/processed/task2/")

    return DatasetDict(splits)


def add_targets_for_task2(example: dict) -> dict:
    '''
    Build a text target for each example from its Quadruplet list.

    Encoding format (simple and parseable):

        Aspect || Opinion || VA
        ### Aspect2 || Opinion2 || VA2
        ...

    Example:
        Quadruplet = [
           {"Aspect": "thai food", "Opinion": "average to good", "VA": "6.75#6.38"},
           {"Aspect": "delivery",  "Opinion": "terrible",       "VA": "2.88#6.62"},
        ]

        target = "thai food || average to good || 6.75#6.38 ### delivery || terrible || 2.88#6.62"

    If no Quadruplet is present (should be rare in train), we emit "NONE".
    '''
    quads = example.get("Quadruplet", []) or []
    segments = []

    for quad in quads:
        aspect = quad.get("Aspect", "NULL")
        opinion = quad.get("Opinion", "NULL")
        va = quad.get("VA", "5.00#5.00")
        segments.append(f"{aspect} || {opinion} || {va}")

    if segments:
        example["target"] = " ### ".join(parts)
    else:
        example["target"] = "NONE"

    return example


# ----Tokenizing ----
def build_preprocess(tokenizer):
    '''
    Return a preprocessing function that:

      - Tokenizes the input Text
      - Tokenizes the target string
      - Attaches labels for seq2seq training
    '''

    def preprocess(batch):
        # Encode inputs
        model_inputs = tokenizer(
            batch["Text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        # Encode targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def main():
    args = parse_args()

    print("[Task2] Loading dataset...")
    dataset = load_task2()

    print("[Task2] Building targets from Quadruplet...")
    dataset = dataset.map(add_targets_for_task2)

    print("[Task2] Loading tokenizer/model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    preprocess = build_preprocess(tokenizer)
    dataset = dataset.map(preprocess, batched=True)

    train_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        eval_strategy="epoch",
        do_eval=True,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="logs_task2",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("valid", None),
        tokenizer=tokenizer,
    )

    print("[Task2] Training seq2seq baseline...")
    trainer.train()

    save_path = args.out_dir / "t5_final"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"[Task2] Saving final model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("[Task2] Done.")


# Main entry point
if __name__ == "__main__":
    main()
