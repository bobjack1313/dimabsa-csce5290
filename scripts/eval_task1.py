#!/usr/bin/env python3
'''
Local evaluator for DimABSA Task 1

Desc:
- Evaluate VA regression
- Ignore Aspect extraction
- Use VA_gold from processed data
'''

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from utils.utils_jsonl import load_jsonl


def eval_task1_local(model, tokenizer, gold_path):
    gold = load_jsonl(gold_path)

    texts = []
    v_gold = []
    a_gold = []

    for ex in gold:
        va = ex.get("VA_gold", {})
        v = va.get("V")
        a = va.get("A")
        if v is None or a is None:
            continue

        texts.append(ex["Text"])
        v_gold.append(float(v))
        a_gold.append(float(a))

    if len(texts) == 0:
        print("ERROR: No gold VA values found.")
        return

    # Tokenize
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()

    v_pred = logits[:, 0]
    a_pred = logits[:, 1]

    # RMSE
    v_rmse = mean_squared_error(v_gold, v_pred) ** 0.5
    a_rmse = mean_squared_error(a_gold, a_pred) ** 0.5
    combined_rmse = mean_squared_error(
        v_gold + a_gold,
        list(v_pred) + list(a_pred)
    ) ** 0.5

    # Pearson
    v_pcc = pearsonr(v_gold, v_pred)[0]
    a_pcc = pearsonr(a_gold, a_pred)[0]

    print("\n=== LOCAL EVALUATION (TASK 1 – Option A) ===")
    print(f"Samples used: {len(texts)}")
    print(f"RMSE (Valence):  {v_rmse:.4f}")
    print(f"RMSE (Arousal):  {a_rmse:.4f}")
    print(f"RMSE (Combined): {combined_rmse:.4f}")
    print(f"PCC  (Valence):  {v_pcc:.4f}")
    print(f"PCC  (Arousal):  {a_pcc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    eval_task1_local(model, tokenizer, args.gold)


if __name__ == "__main__":
    main()
