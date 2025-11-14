#!/usr/bin/env python3
# =============================================================================
# Script Name : prepare_datasets.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
# Description : Merge and normalize the staged DimABSA 2026 datasets for
#               Track A, Subtasks 1 and 2, into unified JSONL files used by
#               our training and evaluation pipeline.
#
# Tasks       :
#   - Task 1 (DimASR): Dimensional Aspect Sentiment Regression
#       - data/processed/task1/train.jsonl
#       - data/processed/task1/valid.jsonl
#
#   - Task 2 (DimASTE): Aspect–Opinion–Valence–Arousal Triplet Extraction
#       - data/processed/task2/train.jsonl
#       - data/processed/task2/valid.jsonl
#
# Expected Inputs (after running stage_data.py):
#   data/raw/task1/
#       eng_restaurant_train_alltasks.jsonl
#       eng_laptop_train_alltasks.jsonl
#       eng_restaurant_dev_task1.jsonl
#       eng_laptop_dev_task1.jsonl
#
#   data/raw/task2/
#       eng_restaurant_train_alltasks.jsonl
#       eng_laptop_train_alltasks.jsonl
#       eng_restaurant_dev_task2.jsonl
#       eng_laptop_dev_task2.jsonl
#
# Notes       :
#   - This script does NOT download anything. It assumes the official
#     DimABSA2026 repo is cloned under external/ and raw files have been
#     staged into data/raw/ by stage_data.py.
#   - Output JSONL schema is intentionally simple and consistent:
#         { "ID": str,
#           "Text": str,
#           "Quadruplet": [
#               { "Aspect": str,
#                 "Category": str,
#                 "Opinion": str,
#                 "VA": str }
#           ]
#         }
#   - VA is kept as a string ("6.75#6.38") when available. If no VA
#     is present in the source, "VA" is set to "NULL".
#
# Author      : Amrit Adhikari, Bob Jack - Group 4
# Date        : 2025-10-29
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from scripts.utils_jsonl import parse_stream_jsonl, write_jsonl


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


# ---- Normalization for Task 1 (DimASR) ----
def normalize_task1_record(raw_rec: Dict) -> Dict:
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

    Logic:
      - If the source already includes "Quadruplet", we clean it.
      - Otherwise we try to build Quadruplet entries from "Aspect" and
        any available VA information. Missing values are filled with
        "NULL" so downstream code never has to guard against None.
    '''
    raw_id = str(raw_rec.get("ID", ""))
    raw_text = raw_rec.get("Text") or raw_rec.get("text") or ""
    # String
    raw_text = str(raw_text)

    # Case 1: full Quadruplet already present with train_alltasks files
    if "Quadruplet" in raw_rec:
        norm_quads = _normalize_quad_list(raw_rec["Quadruplet"])
        return {"ID": raw_id, "Text": raw_text, "Quadruplet": norm_quads}

    # Case 2: dev-style record with Aspect (+/- VA) but no Quadruplet
    aspects = raw_rec.get("Aspect") or []

    if not isinstance(aspects, list):
        if apsects in (None, "NULL"):
            aspects = []
        else:
            [str(aspects)]

    # Try to pick up VA if it exists; structure may vary, so keep it tight
    va_field = raw_rec.get("VA")
    quads_raw = []

    if isinstance(va_field, list) and len(va_field) == len(aspects):
        # Assume VA aligned with Aspect list
        for asp, va in zip(aspects, va_field):
            va_str = va if isinstance(va, str) else str(va)

            quads_raw.append(
                {
                    "Aspect": asp,
                    "Category": raw_rec.get("Category", "NULL"),
                    "Opinion": "NULL",
                    "VA": va_str,
                }
            )
    else:
        # No usable VA alignment; fallback to Aspect only with NULL VA
        for asp in aspects:
            quads_raw.append(
                {
                    "Aspect": asp,
                    "Category": raw_rec.get("Category", "NULL"),
                    "Opinion": "NULL",
                    "VA": "NULL",
                }
            )

    norm_quads = _normalize_quad_list(quads_raw)
    return {"ID": raw_id, "Text": raw_text, "Quadruplet": norm_quads}


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


#---- Combine ----
def merge_split_files(
    source_files: Iterable[Path],
    normalizer: Callable[[Dict], Dict],
    out_path: Path,
) -> None:
    '''
    Merge multiple JSONL files, apply a normalizer to each record, and
    write the unified JSONL file.
    '''
    combined: List[Dict] = []

    for src in source_files:
        if not src.exists():
            print(f"[WARN] Missing source file: {src}")
            continue

        print(f"Reading: {src}")

        # Use stream to parse and combine
        for raw_rec in parse_stream_jsonl(src):
            combined.append(normalizer(raw_rec))

    write_jsonl(combined, out_path)
    print(f"[OK] Wrote {out_path}  ({len(combined)} records)")



def main():
    '''

    '''
    print("Preparing processed datasets for DimABSA 2026 (Task 1 + Task 2)...")

    # Task 1
    merge_split_files(TRAIN_FILES_1, normalize_task1_record, OUT_TASK1 / "train.jsonl")
    merge_split_files(DEV_FILES_1, normalize_task1_record, OUT_TASK1 / "valid.jsonl")

    # Task 2
    merge_split_files(TRAIN_FILES_2, normalize_task2_record, OUT_TASK2 / "train.jsonl")
    merge_split_files(DEV_FILES_2, normalize_task2_record, OUT_TASK2 / "valid.jsonl")

    print("Finished.")



# Entry point to process our filesets
if __name__ == "__main__":
    main()
