#!/usr/bin/env python3
# =============================================================================
# Script Name : prepare_datasets.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
#
# Description :
#   Build unified, normalized JSONL datasets for both DimABSA Track A tasks.
#   This script converts the raw "train_alltasks" files into processed
#   train/valid splits using an 80/20 random partition. These processed
#   files are the official inputs to our training and evaluation pipeline.
#
# Tasks Handled :
#
#   • Task 1 (DimASR) — Dimensional Aspect Sentiment Regression
#       Output files:
#         - data/processed/task1/train.jsonl
#         - data/processed/task1/valid.jsonl
#       Each record contains:
#         - Text
#         - Full Quadruplet list
#         - Extracted VA_gold regression target (V,A)
#
#   • Task 2 (DimASTE) — Aspect–Opinion–Valence–Arousal Extraction
#       Simplified project variant:
#         - Still uses VA-bearing train_alltasks files
#         - Produces parallel train/valid splits for Task 2 modeling
#       Output files:
#         - data/processed/task2/train.jsonl
#         - data/processed/task2/valid.jsonl
#
# Expected Inputs (after running stage_data.py):
#
#   data/raw/task1/
#       eng_laptop_train_alltasks.jsonl
#       eng_restaurant_train_alltasks.jsonl
#
#   data/raw/task2/
#       eng_laptop_train_alltasks.jsonl
#       eng_restaurant_train_alltasks.jsonl
#
#   NOTE:
#       • Dev files from the official dataset do NOT contain Quadruplets
#         or VA labels, so they cannot be used for validation.
#       • This script intentionally ignores all dev files for both tasks.
#
# Output Format :
#     {
#       "ID": str,
#       "Text": str,
#       "Quadruplet": [
#           { "Aspect": str, "Category": str,
#             "Opinion": str, "VA": "7.25#6.87" }
#       ],
#       "VA_gold": { "V": float or None, "A": float or None }
#     }
#
# Notes :
#   • This script does not download data.
#   • It assumes stage_data.py has already copied raw files to data/raw/.
#   • Normalization is task-specific: Task 1 extracts VA_gold; Task 2
#     keeps full Quadruplets.
#
# Author      : Amrit Adhikari, Bob Jack — Group 4
# Last Updated: 2025-11-16
# =============================================================================

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from utils.utils_jsonl import parse_stream_jsonl, write_jsonl

'''
Dataset setups
Format A:
Text
Aspect

Format B:
Text
Quadruplet: [
    {"Aspect": "...", "VA": "4.33#5.12"}
]
----------------
Task 1 Raw:
"Quadruplet": [
  {
    "Aspect": "unit",
    "Category": "LAPTOP#DESIGN_FEATURES",
    "Opinion": "pretty",
    "VA": "7.12#7.12"
  }
]

'''

# --- Source and Destination File Paths ---
ROOT = Path(__file__).resolve().parent.parent

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

# Dev files with both restuarant and laptop domains
DEV_FILES_1= [
    SRC_TASK1 / "eng_laptop_dev_task1.jsonl",
    SRC_TASK1 / "eng_restaurant_dev_task1.jsonl",
]

DEV_FILES_2= [
    SRC_TASK2 / "eng_laptop_dev_task2.jsonl",
    SRC_TASK2 / "eng_restaurant_dev_task2.jsonl",
]


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
    Normalize a raw Task 1 record into:

        { "ID": str,
          "Text": str,
          "Quadruplet": [
              { "Aspect": str,
                "Category": str,
                "Opinion": str,
                "VA": str }
          ]
        }

    For DimASR (Task 1), each record may contain multiple quadruplets,
    each with its own VA label.

    Strategy is to use only the first quadruplet's VA as the regression target.

    Logic:
      - If the source already includes "Quadruplet", we clean it.
      - Otherwise we try to build Quadruplet entries from "Aspect" and
        any available VA information. Missing values are filled with
        "NULL" so downstream code never has to guard against None.
    '''

    text = raw_rec.get("Text") or ""
    quads = raw_rec.get("Quadruplet") or []

    # aspects list (only Aspect strings)
    aspects = []

    for quad in quads:
        # Extract the Aspect field
        aspect_value = quad.get("Aspect")

        # Skip if missing or NULL
        if aspect_value is None:
            continue
        if aspect_value == "NULL":
            continue
        if aspect_value == "":
            continue

        # Otherwise, keep it
        aspects.append(aspect_value)

    # extract VA from first quad (all quads share the same VA)
    v = a = None
    if quads:
        va_str = quads[0].get("VA")
        if isinstance(va_str, str) and "#" in va_str:
            try:
                v, a = map(float, va_str.split("#"))
            except Exception:
                pass

    return {
        "ID": raw_rec.get("ID", ""),
        "Text": text,
        "Aspect": aspects,
        "VA_gold": {"V": v, "A": a},
        "Quadruplet": quads
    }


# ---- Normalization for Task 2 (DimASTE) ----
def normalize_task2_record(raw_rec: Dict) -> Dict:
    '''
    Normalize a raw Task 2 record into the same Quadruplet-based schema.

    Task 2 extracts (Aspect, Opinion, Category, VA) tuples from text.
    The official files should already contain a 'Quadruplet'
    field, but we still clean it to be safe.
    '''
    raw_id = str(raw_rec.get("ID", ""))
    raw_text = raw_rec.get("Text") or raw_rec.get("text") or ""
    raw_text = str(raw_text)

    norm_quads = _normalize_quad_list(raw_rec.get("Quadruplet"))
    return {"ID": raw_id, "Text": raw_text, "Quadruplet": norm_quads}


# ---- Splits ----
def build_task_splits(
    in_files: Iterable[Path],
    out_dir: Path,
    normalizer: Callable[[Dict], Dict],
    task_name: str = "Task"
):
    '''
    Build an 80/20 train–valid split for any DimABSA task.

    Parameters
    ----------
    in_files : Iterable[Path]
        A list of raw JSONL files that contain VA-bearing Quadruplet annotations.
        (For Task 1: train_alltasks files. For Task 2: train_alltasks files.)

    out_dir : Path
        Output directory where train.jsonl and valid.jsonl will be written.

    normalizer : Callable
        A function that converts one raw JSON object into the normalized
        project format:
            { "ID", "Text", "Quadruplet": [...], ... }

    task_name : str
        Name of the task (e.g., "Task1", "Task2") used only for printing messages.
    '''

    # Init all recs
    all_recs: List[Dict] = []

    print(f"\n[{task_name}] Loading raw annotated records...")

    # Loop the in files to run stream parser
    for src in in_files:
        print(f"[{task_name}] Reading: {src}")

        for raw in parse_stream_jsonl(src):
            # Normalize the raw JSON
            normalized = normalizer(raw)
            all_recs.append(normalized)

    print(f"[{task_name}] Loaded {len(all_recs)} total records.")

    if not all_recs:
        print(f"[{task_name}] ERROR: No records found. Check input files.")
        return

    # Shuffle the deck
    random.shuffle(all_recs)

    # Split 80/20
    n = len(all_recs)
    split_index = int(n * 0.80)

    train_recs = all_recs[:split_index]
    valid_recs = all_recs[split_index:]

    print(f"[{task_name}] Train split: {len(train_recs)} records")
    print(f"[{task_name}] Valid split: {len(valid_recs)} records")

    # Write out
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    write_jsonl(train_recs, train_path)
    write_jsonl(valid_recs, valid_path)

    print(f"[{task_name}] Wrote:")
    print(f"   - {train_path}")
    print(f"   - {valid_path}")
    print(f"[{task_name}] Completed train/valid split.\n")


def main():
    print("Preparing processed datasets for DimABSA 2026 (Task 1 + Task 2)...")

    # Task 1
    build_task_splits(
        in_files=TRAIN_FILES_1,
        out_dir=OUT_TASK1,
        normalizer=normalize_task1_record,
        task_name="Task1"
    )

    # Task 2
    build_task_splits(
        in_files=TRAIN_FILES_2,
        out_dir=OUT_TASK2,
        normalizer=normalize_task2_record,
        task_name="Task2"
    )

    print("Finished.")


# Entry point to process our filesets
if __name__ == "__main__":
    main()
