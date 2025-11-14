#!/usr/bin/env python
# =============================================================================
# Script Name : train.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
# Description : Train a BERT-based regression model for DimABSA Task 1 (DimASR).
#
# Overview:
#   - Loads pre-processed JSONL datasets from data/processed/<task>/.
#   - Normalizes records into a consistent HuggingFace DatasetDict.
#   - Tokenizes text inputs using a selected Transformer model.
#   - Extracts VA (Valence, Arousal) values as 2-dimensional regression labels.
#   - Trains a HuggingFace regression model using Trainer API.
#
# Command-line Arguments:
#   --task        : task1 (default) or task2 — determines which folder to load.
#   --model       : HuggingFace model name (default: bert-base-uncased)
#   --epochs      : number of training epochs (default: 3)
#   --batch-size  : batch size per device (default: 8)
#   --lr          : learning rate (default: 5e-5)
#   --out-dir     : where to save checkpoints (default: experiments/checkpoints)
#
# Notes:
#   - This script assumes that stage_data.py and prepare_datasets.py have already
#     been executed to generate the processed JSONL files.
#   - Task 1 and Task 2 currently share the same pipeline structure, but Task 2
#     may be extended with its own VA-label strategy.
#   - The Trainer loop can be mocked in unit tests; see tests/test_train.py.
#
# Authors     : Amrit Adhikari, Bob Jack - Group 4
# Date        : 2025-11-12
# =============================================================================

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def parse_args():
    '''
    Parse command-line arguments for the training pipeline.

    Returns:
        argparse.Namespace: Parsed arguments including:
            task        – Which dataset/task to train on ("task1" or "task2").
            model       – HuggingFace model identifier to load.
            epochs      – Number of training epochs.
            batch_size  – Batch size per device.
            lr          – Learning rate.
            out_dir     – Directory where checkpoints will be saved.

    Notes:
        - This function only defines configuration; actual training logic is
          implemented in main().
    '''
    p = argparse.ArgumentParser(description="Train BERT model for DimABSA Task 1 (DimASR)")
    p.add_argument("--task", choices=["task1", "task2"], default="task1")
    p.add_argument("--model", type=str, default="bert-base-uncased", help="HuggingFace model name")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/checkpoints"))

    return p.parse_args()


def load_local_dataset(task: str):
    '''
    Load and normalize the processed DimABSA dataset for a given task.

    This function reads JSONL files from data/processed/<task>/ and constructs
    a HuggingFace DatasetDict with "train" and (optionally) "valid" splits.
    Records are normalized so downstream tokenization and training do not need
    to handle inconsistent schema from different source files.

    Normalization rules:
        - If a record includes "Quadruplet", all Aspect values are extracted
          and mirrored into an "Aspect" list.
        - If a record only provides "Aspect", synthetic Quadruplet entries are
          created with empty Category/Opinion/VA fields.
        - If neither is present, Aspect and Quadruplet fields default to [].

    Args:
        task (str): The task identifier ("task1" or "task2").

    Returns:
        DatasetDict: A dictionary-like object with HuggingFace Dataset splits.

    Raises:
        FileNotFoundError: If no JSONL files are found for the requested task.

    Notes:
        - This function performs only data loading and structural normalization.
        - Tokenization and label construction are handled separately in main().
        - Tests use temporary directories to validate all branches of this logic.
    '''
    base_dir = Path("data/processed") / task
    splits = {}

    for split in ["train", "valid"]:
        path = base_dir / f"{split}.jsonl"

        if not path.exists():
            continue

        rows = []
        with open(path, "r", encoding="utf-8") as file:

            for line in file:
                example = json.loads(line)

                # Normalize structure
                if "Quadruplet" in example:
                    # Extract all aspect values from the Quadruplet list
                    aspects = []
                    for quad in example["Quadruplet"]:
                        if isinstance(quad, dict):
                            aspect_value = quad.get("Aspect")
                            if aspect_value is not None and aspect_value != "":
                                aspects.append(aspect_value)

                    # Store extracted aspects back into the example
                    example["Aspect"] = aspects

                elif "Aspect" in example:
                    # Build a new Quadruplet list from existing Aspect entries
                    quadruplets = []

                    for asp in example.get("Aspect", []):
                        # Create a full Quadruplet entry with empty placeholders
                        new_quad = {
                            "Aspect": asp,
                            "Category": "",
                            "Opinion": "",
                            "VA": ""
                        }
                        quadruplets.append(new_quad)

                    # Store the synthetic Quadruplet list
                    example["Quadruplet"] = quadruplets

                else:
                    # If neither Aspect nor Quadruplet exists, create both as empty lists
                    example["Aspect"] = []
                    example["Quadruplet"] = []

                rows.append(example)

        splits[split] = Dataset.from_list(rows)

    if not splits:
        raise FileNotFoundError(f"No JSONL data found under {base_dir}")

    return DatasetDict(splits)


def preprocess_factory(tokenizer):
    '''
    Returns a preprocess function bound to the provided tokenizer.
    This avoids using a closure inside main().
    '''
    def preprocess(batch):
        return tokenizer(
            batch["Text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    return preprocess


def parse_va(example):
    '''
    Extracts VA regression targets from the example dict.
    If VA is missing, assigns a neutral fallback label tensor.
    '''
    if "Aspect_VA" in example:
        va_string = example["Aspect_VA"][0]["VA"]
        parts = va_string.split("#")

        first_value = float(parts[0])
        second_value = float(parts[1])

        example["labels"] = torch.tensor([first_value, second_value])

    else:
        # Default neutral values
        example["labels"] = torch.tensor([5.0, 5.0])

    return example


def main():
    args = parse_args()
    dataset = load_local_dataset(args.task)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build a preprocess function bound to this tokenizer
    preprocess = preprocess_factory(tokenizer)

    # Apply preprocessing
    dataset = dataset.map(preprocess, batched=True)

    # Apply VA-label extraction
    dataset = dataset.map(parse_va)

    # Setup model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        problem_type="regression",
    )

    training_args = TrainingArguments(
        output_dir=str(args.out_dir / args.task),
        eval_strategy="epoch",
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
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("valid", None),
        tokenizer=tokenizer,
    )

    # Train the beast
    trainer.train()
    save_path = args.out_dir / args.task / "bert_final"

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model saved to {save_path}")


# Entry point
if __name__ == "__main__":
    main()
