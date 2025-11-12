#!/usr/bin/env python3
"""
# =============================================================================
# Script Name : stage_data.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
# Description : Merges and cleans the official DimABSA Task 1 (DimASR) dataset
#               from multiple domain JSONL files (Restaurant and Laptop)
#               into a unified training and validation dataset.
#
# Task        : Subtask 1 - Dimensional Aspect Sentiment Regression (DimASR)
# Author      : Amrit Adhikari, Bob Jack - Group 4
# Date        : 2025-10-29
# =============================================================================
#

Stage raw DimABSA 2026 dataset files into the local data pipeline.

This script copies the English subtask files from the external
DimABSA2026 repository into the project's data directory. Preprocessing prep
is performed for the two tasks.

Expected input:
    external/DimABSA2026/task-dataset/track_a/subtask_1/eng/

Expected output:
    data/raw/task1/train.jsonl
    data/raw/task1/valid.jsonl

Usage:
    python stage_data.py

Notes:
    - This script does not download from the internet. Must have side repo.
    - It verifies source file presence and checks for naming.
    - Safe to rerun. Existing files are overwritten with new.

"""
from __future__ import annotations
import json
from pathlib import Path


# File Paths
ROOT = Path(__file__).resolve().parent.parent
SRC_BASE = ROOT / "external" / "DimABSA2026" / "task-dataset" / "track_a"
OUT_BASE = ROOT / "data" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    '''
    Load a JSON Lines (.jsonl) file into memory.

    Each line in the file should contain a valid JSON object.
    Blank lines are skipped. Invalid lines are reported but ignored.

    Args:
        path (Path): Path to the .jsonl file to load.

    Returns:
        list[dict]: A list of parsed JSON objects.
    '''

    samples = []
    with path.open("r", encoding="utf-8") as open_file:

        for line in open_file:
            line = line.strip()

            if not line:
                # Skip empty lines
                continue
            try:
                # Parse each line as an independent JSON object
                samples.append(json.loads(line))
            except json.JSONDecodeError as ex:
                # Log a warning and continue on malformed lines
                print(f"[WARN] Skipping bad JSON line in {path.name}: {ex}")
    return samples



def write_jsonl(samples: list[dict], out_file: Path) -> None:
    '''
    Write a list of JSON-serializable objects to a JSON Lines (.jsonl) file.

    Each dictionary in `samples` is serialized to a single line of JSON.
    The parent directory is created automatically if it does not exist.

    Args:
        samples (list[dict]): List of JSON-serializable objects to write.
        out_file (Path): Destination path for the .jsonl file.

    Returns:
        None
    '''
    # Check dir and file exists
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Open file for write
    with out_file.open("w", encoding="utf-8") as ofile:
        for obj in samples:
            ofile.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Display confirmation
    print(f"Wrote {out_file}  ({len(samples)} records)")


