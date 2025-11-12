#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def parse_args():
    p = argparse.ArgumentParser(description="Train BERT model for DimABSA Task 1 (DimASR)")
    p.add_argument("--task", choices=["task1", "task2"], default="task1")
    p.add_argument("--model", type=str, default="bert-base-uncased", help="HuggingFace model name")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/checkpoints"))
    return p.parse_args()


# def load_local_dataset(task):
#     data_files = {
#         "train": f"data/processed/{task}/train.jsonl",
#         "validation": f"data/processed/{task}/valid.jsonl",
#     }
#     return load_dataset("json", data_files=data_files)


# def load_local_dataset(task: str):
#     """Load and normalize dataset files for the given task."""
#     base = Path("data/processed") / task
#     data_files = {
#         "train": str(base / "train.jsonl"),
#         "validation": str(base / "valid.jsonl"),
#     }
#     ds = load_dataset("json", data_files=data_files)

#     # Normalize fields: for consistent columns
#     def normalize(example):
#         if "Quadruplet" in example:
#             # Flatten Quadruplet into Aspect list for compatibility
#             aspects = [q.get("Aspect") for q in example["Quadruplet"] if q.get("Aspect")]
#             example["Aspect"] = aspects
#         if "Quadruplet" not in example:
#             example["Quadruplet"] = [
#                 {"Aspect": a, "Category": "", "Opinion": "", "VA": ""}
#                 for a in example.get("Aspect", [])
#             ]
#         return example

#     ds = ds.map(normalize)
#     return ds



# def load_local_dataset(task: str):
#     """Load and normalize dataset files for the given task."""
#     base = Path("data/processed") / task
#     data_files = {
#         "train": str(base / "train.jsonl"),
#         "validation": str(base / "valid.jsonl"),
#     }
#     ds = load_dataset("json", data_files=data_files)

#     # Normalize fields: ensure consistent columns
#     def normalize(example):
#         if "Quadruplet" in example:
#             # Flatten Quadruplet into Aspect list for compatibility
#             aspects = [q.get("Aspect") for q in example["Quadruplet"] if q.get("Aspect")]
#             example["Aspect"] = aspects
#         if "Quadruplet" not in example:
#             example["Quadruplet"] = [
#                 {"Aspect": a, "Category": "", "Opinion": "", "VA": ""}
#                 for a in example.get("Aspect", [])
#             ]
#         return example

#     ds = ds.map(normalize)
#     return ds


def load_local_dataset(task: str):
    """Custom JSON loader that normalizes DimABSA data before creating HF Dataset."""
    base = Path("data/processed") / task
    splits = {}
    for split in ["train", "valid"]:
        path = base / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                # Normalize structure
                if "Quadruplet" in ex:
                    aspects = [q.get("Aspect") for q in ex["Quadruplet"] if q.get("Aspect")]
                    ex["Aspect"] = aspects
                elif "Aspect" in ex:
                    # Create synthetic Quadruplet entries for uniformity
                    ex["Quadruplet"] = [
                        {"Aspect": a, "Category": "", "Opinion": "", "VA": ""}
                        for a in ex.get("Aspect", [])
                    ]
                else:
                    ex["Aspect"] = []
                    ex["Quadruplet"] = []
                rows.append(ex)
        splits[split] = Dataset.from_list(rows)

    if not splits:
        raise FileNotFoundError(f"No JSONL data found under {base}")

    return DatasetDict(splits)



def main():
    args = parse_args()
    ds = load_local_dataset(args.task)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(batch):
        return tokenizer(batch["Text"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(preprocess, batched=True)

    # For now, use dummy numeric labels (regression target) from VA strings
    def parse_va(example):
        if "Aspect_VA" in example:
            # Take first VA as a rough regression target
            va = example["Aspect_VA"][0]["VA"].split("#")
            v = float(va[0])
            a = float(va[1])
            example["labels"] = torch.tensor([v, a])
        else:
            example["labels"] = torch.tensor([5.0, 5.0])  # neutral fallback
        return example

    ds = ds.map(parse_va)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        problem_type="regression",
    )

    # training_args = TrainingArguments(
    #     output_dir=str(args.out_dir / args.task),
    #     evaluation_strategy="epoch",
    #     learning_rate=args.lr,
    #     per_device_train_batch_size=args.batch_size,
    #     num_train_epochs=args.epochs,
    #     weight_decay=0.01,
    #     save_total_limit=2,
    #     logging_dir="logs",
    #     logging_steps=50,
    # )

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
        train_dataset=ds["train"],
        eval_dataset=ds.get("valid", None),
        tokenizer=tokenizer,
    )

    trainer.train()
    save_path = args.out_dir / args.task / "bert_final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
