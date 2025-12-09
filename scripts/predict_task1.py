#!/usr/bin/env python3
'''
Generate official-format predictions for DimABSA Task 1 (DimASR).

INPUT:
    A JSONL file with fields:
        {
            "ID": "...",
            "Text": "...",
            "Aspect": ["battery", "screen", ...]
        }

OUTPUT:
    A JSONL file with official Codabench format:
        {
            "ID": "...",
            "Aspect_VA": [
                {"Aspect": "battery", "VA": "6.75#7.12"},
                {"Aspect": "screen",  "VA": "5.88#6.00"}
            ]
        }

REQUIREMENTS:
    - A trained regression checkpoint:
        experiments/checkpoints/task1/bert_final/
'''
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys

# Check that project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.utils_jsonl import load_jsonl,save_jsonl,format_va


def predict_for_aspect(text, model, tokenizer):
    '''
    Run model on the full text. Aspect is NOT appended
    '''
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()[0]

    v, a = logits.tolist()
    return v, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path,
                    default=Path("experiments/checkpoints/task1/bert_final"))
    ap.add_argument("--input", type=Path, required=True,
                    help="Path to JSONL input file")
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to write predictions")
    args = ap.parse_args()

    # Load model and tokenizer
    # Always load tokenizer from the original pretrained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    rows_out = []

    for sample in load_jsonl(args.input):
        sid = sample["ID"]
        text = sample["Text"]
        aspects = sample.get("Aspect", [])

        aspect_va = []

        for asp in aspects:
            v, a = predict_for_aspect(text, model, tokenizer)
            va_str = format_va(v, a)

            aspect_va.append({
                "Aspect": asp,
                "VA": va_str
            })

        rows_out.append({
            "ID": sid,
            "Aspect_VA": aspect_va
        })

    save_jsonl(rows_out, args.output)
    print(f"[OK] Wrote predictions to {args.output}")


# Main entry point
if __name__ == "__main__":
    main()
