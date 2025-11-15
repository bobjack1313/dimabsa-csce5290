#!/usr/bin/env python3
'''
Evaluation script for both tasks.

Supports:

• Task 1 (DimASR):
      - Regression model predicts [Valence, Arousal]
      - Metrics: RMSE + Pearson Correlation

• Task 2 (DimASTE – Simplified):
      - Regression model predicts triplet count
      - Metric: RMSE against gold count

Usage:
    python -m scripts.eval --task task1
    python -m scripts.eval --task task2


'''

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.utils_jsonl import load_jsonl


# Task 1 (DimASR) — Regression: Valence + Arousal
def eval_task1(model, tokenizer, valid_path: Path):
    samples = load_jsonl(valid_path)

    texts = []
    v_gold = []
    a_gold = []

    for ex in samples:
        quads = ex.get("Quadruplet", [])
        if not quads:
            continue

        va_str = quads[0].get("VA")
        if not va_str or "#" not in va_str:
            continue

        try:
            v, a = map(float, va_str.split("#"))
        except:
            continue

        texts.append(ex["Text"])
        v_gold.append(v)
        a_gold.append(a)

    if not texts:
        print("[ERROR] No valid VA labels found in validation file.")
        return

    # Tokenize
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():
        logits = model(**batch).logits.cpu().numpy()

    v_pred = logits[:, 0]
    a_pred = logits[:, 1]

    # Metrics
    v_mse = mean_squared_error(v_gold, v_pred)
    a_mse = mean_squared_error(a_gold, a_pred)

    v_rmse = v_mse ** 0.5
    a_rmse = a_mse ** 0.5

    v_pcc, _ = pearsonr(v_gold, v_pred)
    a_pcc, _ = pearsonr(a_gold, a_pred)

    print("\n=== Task 1 Evaluation (DimASR Regression) ===")
    print(f"Samples evaluated: {len(texts)}")
    print(f"Valence RMSE: {v_rmse:.4f}")
    print(f"Arousal RMSE: {a_rmse:.4f}")
    print(f"Valence PCC:  {v_pcc:.4f}")
    print(f"Arousal PCC:  {a_pcc:.4f}")


def eval_task2(model, tokenizer, valid_path: Path):
    samples = load_jsonl(valid_path)

    texts = []
    v_gold = []
    a_gold = []

    for ex in samples:
        quads = ex.get("Quadruplet", [])
        if not quads:
            continue

        va_str = quads[0].get("VA")
        if not va_str or "#" not in va_str:
            continue

        try:
            v, a = map(float, va_str.split("#"))
        except:
            continue

        texts.append(ex["Text"])
        v_gold.append(v)
        a_gold.append(a)

    if not texts:
        print("[ERROR] No valid VA labels found in Task 2 validation file.")
        return

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**batch).logits.cpu().numpy()

    v_pred = logits[:, 0]
    a_pred = logits[:, 1]

    v_mse = mean_squared_error(v_gold, v_pred)
    a_mse = mean_squared_error(a_gold, a_pred)

    v_rmse = v_mse ** 0.5
    a_rmse = a_mse ** 0.5

    v_pcc, _ = pearsonr(v_gold, v_pred)
    a_pcc, _ = pearsonr(a_gold, a_pred)

    print("\n=== Task 2 Evaluation (DimASTE – VA Regression) ===")
    print(f"Samples evaluated: {len(texts)}")
    print(f"Valence RMSE: {v_rmse:.4f}")
    print(f"Arousal RMSE: {a_rmse:.4f}")
    print(f"Valence PCC:  {v_pcc:.4f}")
    print(f"Arousal PCC:  {a_pcc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["task1", "task2"], required=True)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--data-dir", type=Path)
    args = ap.parse_args()

    # Default paths
    if args.model_dir is None:
        args.model_dir = Path(f"experiments/checkpoints/{args.task}/bert_final")
    if args.data_dir is None:
        args.data_dir = Path(f"data/processed/{args.task}/valid.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    # Dispatch
    if args.task == "task1":
        eval_task1(model, tokenizer, args.data_dir)
    else:
        eval_task2(model, tokenizer, args.data_dir)


# Entry Point
if __name__ == "__main__":
    main()



'''
Improvements for task 2

We do one of these:

 - first VA only

    Then best-case PCC ~0.4 – 0.6
    RMSE ~1.0 – 1.5


- average VA of all quad labels

    Then PCC usually lower
    RMSE improves slightly
    But the model becomes more stable


- VA_gold = first quadruplet VA
or Task-2 VA_gold = mean of all quad VAs

 - train one sample per quadruplet (May give best scores)

'''
