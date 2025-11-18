#!/usr/bin/env python3
# =============================================================================
# Script Name : prepare_datasets.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
# Description :
#   Build clean train / validation splits for Track A, Subtasks 1 and 2
#   using ONLY the official *train_alltasks* files (which contain gold VA).
#
#   - We DO NOT use the dev files for splitting, since they do not contain
#     VA / Quadruplet labels. They are input-only evaluation sets.
#
# Outputs:
#   Task 1 (DimASR – regression)
#       data/processed/task1/train.jsonl
#       data/processed/task1/valid.jsonl
#
#   Task 2 (DimASTE – extraction)
#       data/processed/task2/train.jsonl
#       data/processed/task2/valid.jsonl
#
#   Each output file is a shuffled 80/20 split of the corresponding
#   *_train_alltasks.jsonl files for laptop + restaurant.
#
# Schema (Task 1):
#   {
#     "ID": str,
#     "Text": str,
#     "Aspect": [str, ...],          # list of aspect strings (may be empty)
#     "VA_gold": {"V": float|None,
#                 "A": float|None},  # sentence-level VA used as regression target
#     "Quadruplet": [
#         {"Aspect": str,
#          "Category": str,
#          "Opinion": str,
#          "VA": str}
#     ]
#   }
#
# Schema (Task 2):
#   {
#     "ID": str,
#     "Text": str,
#     "Quadruplet": [
#         {"Aspect": str,
#          "Category": str,
#          "Opinion": str,
#          "VA": str}
#     ]
#   }
#
# Notes:
#   - For Task 1, we keep the full Quadruplet list but also derive a single
#     VA_gold value by taking the VA from the *first* quadruplet, when present.
#     This is the regression target used in our current model.
#
#   - For Task 2, we do not touch the Quadruplet labels; we simply normalize
#     them into a consistent list-of-dicts structure.
#
# Author      : Amrit Adhikari, Bob Jack — Group 4
# Last Updated: 2025-11-17
# =============================================================================
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils_jsonl import parse_stream_jsonl, write_jsonl

# '''
# Dataset setups
# Format A:
# Text
# Aspect

# Format B:
# Text
# Quadruplet: [
#     {"Aspect": "...", "VA": "4.33#5.12"}
# ]
# ----------------
# Task 1 Raw:
# "Quadruplet": [
#   {
#     "Aspect": "unit",
#     "Category": "LAPTOP#DESIGN_FEATURES",
#     "Opinion": "pretty",
#     "VA": "7.12#7.12"
#   }
# ]

# '''

# --- Source and Destination File Paths ---
SRC_TASK1 = ROOT / "data" / "raw" / "task1"
SRC_TASK2 = ROOT / "data" / "raw" / "task2"

OUT_TASK1 = ROOT / "data" / "processed" / "task1"
OUT_TASK2 = ROOT / "data" / "processed" / "task2"
OUT_TASK1.mkdir(parents=True, exist_ok=True)
OUT_TASK2.mkdir(parents=True, exist_ok=True)

# Train files with both restuarant and laptop domains
TRAIN_FILES_1 = [
    SRC_TASK1 / "eng_laptop_train_alltasks.jsonl",
    SRC_TASK1 / "eng_restaurant_train_alltasks.jsonl",
]

TRAIN_FILES_2 = [
    SRC_TASK2 / "eng_laptop_train_alltasks.jsonl",
    SRC_TASK2 / "eng_restaurant_train_alltasks.jsonl",
]


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---- Helpers ----
def _normalize_quad_list(raw_quads) -> List[Dict[str, str]]:
    '''
    Internal utility:
    There are Quadruplet dicts that need to be normalized into
    a list of {Aspect, Category, Opinion, VA} strings.
    '''
    # Return if not applicable
    if not isinstance(raw_quads, list):
        return []

    # Initialize normalized
    normalized: List[Dict[str, str]] = []

    # Loop through and create lists
    for quad in raw_quads:
        if not isinstance(quad, dict):
            continue

        # Init with NULL
        aspect = quad.get("Aspect", "NULL")
        category = quad.get("Category", "NULL")
        opinion = quad.get("Opinion", "NULL")
        va = quad.get("VA", "NULL")

        # VAs can appear odd, they get flattened here
        if isinstance(va, (list, dict)):
            va = str(va)

        # Put into dict
        normalized.append(
            {
                "Aspect": str(aspect),
                "Category": str(category),
                "Opinion": str(opinion),
                "VA": str(va),
            }
        )

    return normalized


def normalize_task1_record(raw_rec: dict) -> dict:
    '''
    Normalize a raw Task 1 record from and to below:

    Input (from *_train_alltasks.jsonl):
        {
          "ID": ...,
          "Text": ...,
          "Quadruplet": [
              {"Aspect": "...", "Category": "...", "Opinion": "...", "VA": "V#A"},
              ...
          ]
        }

    Output:
        {
          "ID": ...,
          "Text": ...,
          "Aspect": [aspect_1, aspect_2, ...],
          "VA_gold": {"V": float|None, "A": float|None},
          "Quadruplet": [normalized quadruplet dicts...]
        }

    Notes:
      - We keep all Quadruplets.
      - We derive a single VA_gold by taking the VA from the first Quadruplet.
        This matches the current regression model, which predicts one (V, A)
        pair per sentence.
    For DimASR (Task 1), each record may contain multiple quadruplets,
    each with its own VA label.
    '''
    text = raw_rec.get("Text") or ""
    quads = _normalize_quad_list(raw_rec.get("Quadruplet"))

    # Extract aspect list
    aspects: List[str] = []
    for quad in quads:
        asp = quad.get("Aspect")
        if asp and asp != "NULL":
            aspects.append(asp)

    # If there are quadruplets, get VA_gold
    v = None
    a = None

    if quads:
        va_str = quads[0].get("VA")
        if isinstance(va_str, str) and "#" in va_str:
            try:
                v, a = map(float, va_str.split("#"))
            except Exception:
                v = None
                a = None

    return {
        "ID": raw_rec.get("ID", ""),
        "Text": text,
        "Aspect": aspects,
        "VA_gold": {"V": v, "A": a},
        "Quadruplet": quads
    }


def normalize_task2_record(raw_rec: Dict) -> Dict:
    '''
    Normalize one Task 2 (DimASTE) training record.

    Input (from *_train_alltasks.jsonl):
        {
          "ID": ...,
          "Text": ...,
          "Quadruplet": [...]
        }

    Output:
        {
          "ID": ...,
          "Text": ...,
          "Quadruplet": [normalized quadruplet dicts...]
        }

    We do NOT convert to Triplets here. We keep full Quadruplets so that
    downstream Task 2 components can decide how to use them.
    '''
    raw_id = str(raw_rec.get("ID", ""))
    raw_text = raw_rec.get("Text") or raw_rec.get("text") or ""
    raw_text = str(raw_text)

    norm_quads = _normalize_quad_list(raw_rec.get("Quadruplet"))
    return {"ID": raw_id, "Text": raw_text, "Quadruplet": norm_quads}


# ---- Splits ----
def build_task_split(
    in_files: Iterable[Path],
    normalizer: Callable[[Dict], Dict],
    out_dir: Path,
    task_name: str,
    seed: int = 42,
    train_filename: str = "train.jsonl",
    valid_filename: str = "valid.jsonl",
) -> None:
    '''
    Build an 80/20 train–valid split for any DimABSA task.

    Parameters
    ----------
    in_files : Iterable[Path]
        A list of raw JSONL files that contain VA-bearing Quadruplet annotations.
        (For Task 1: train_alltasks files. For Task 2: train_alltasks files.)

    out_dir : Path
        Output directory where train.jsonl and valid.jsonl will be written.

    task_name: String to dictate which task to operate

    seed: Int value for reproducability

    normalizer : Callable
        A function that converts one raw JSON object into the normalized
        project format:
            { "ID", "Text", "Quadruplet": [...], ... }

    train_filename: Strin for the trianing file in case it needs to change.

    valid_filenaame: String for the training file for validation in case it needs to change.
    '''

    # Init all recs
    all_recs: List[Dict] = []

    print(f"[{task_name}] Loading raw annotated records...")

    # Loop the in files to run stream parser
    for src in in_files:
        if not src.exists():
            print(f"[{task_name}] [WARN] Missing source file: {src}")
            continue

        print(f"[{task_name}] Reading: {src}")

        for raw in parse_stream_jsonl(src):
            # Normalize the raw JSON
            normalized = normalizer(raw)
            all_recs.append(normalized)

    print(f"[{task_name}] Loaded {len(all_recs)} total records.")

    rec_total = len(all_recs)
    if rec_total == 0:
        print(f"[{task_name}] [ERROR] No records loaded; aborting split.")
        return

    # Shuffle the deck
    random.seed(seed)
    random.shuffle(all_recs)

    # Split 80/20
    split_idx = int(rec_total * 0.8)

    train_recs = all_recs[:split_idx]
    valid_recs = all_recs[split_idx:]

    print(f"[{task_name}] Total records: {rec_total}")
    print(f"[{task_name}] Train: {len(train_recs)} | Valid: {len(valid_recs)}")

    # Write out
    write_jsonl(train_recs, out_dir / train_filename)
    write_jsonl(valid_recs, out_dir / valid_filename)

    print(f"[{task_name}] Wrote:")
    print(f"   - {train_filename}")
    print(f"   - {valid_filename}")
    print(f"[{task_name}] Completed train/valid split.\n")


def main():
    print("Preparing processed datasets for DimABSA 2026 (Task 1 + Task 2)...")

    # Task 1
    build_task_split(
        in_files=TRAIN_FILES_1,
        normalizer=normalize_task1_record,
        out_dir=OUT_TASK1,
        task_name="Task1-DimASR",
    )

    print()

    # Task 2
    build_task_split(
        in_files=TRAIN_FILES_2,
        normalizer=normalize_task2_record,
        out_dir=OUT_TASK2,
        task_name="Task2-DimASTE",
    )

    print("\n[OK] Finished building processed train/valid splits.")


# Entry point to process our filesets
if __name__ == "__main__":
    main()
