#!/usr/bin/env python3
# =============================================================================
# Script Name : utils_jsonl.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
# Description : Utility functions for working with JSONL datasets. Provides
#               safe, reusable helpers for reading, writing, and validating
#               JSON Lines files used throughout the DimABSA preprocessing
#               and training pipeline.
#
# Functionality:
#   - load_jsonl(path):    Read line-delimited JSON into Python dictionaries.
#   - write_jsonl(items):  Write Python dictionaries back to JSONL format.
#   - Basic validation and error reporting for malformed lines.
#
# Used by     : stage_data.py, prepare_task1.py, prepare_task2.py,
#               training and evaluation scripts.
#
# Notes       : - These helpers perform no dataset-specific logic.
#               - All normalization/merging logic belongs in prepare_*.py.
#               - Keeping JSONL helpers isolated avoids duplication across
#                 the pipeline and supports cleaner unit testing.
#
# Author      : Amrit Adhikari, Bob Jack - Group 4
# Date        : 2025-10-29
# =============================================================================
import json
from pathlib import Path
from typing import Iterable, Dict


def parse_stream_jsonl(path: Path) -> Iterable[Dict]:
    '''
    Load a JSON Lines (.jsonl) file into memory.

    Each line in the file should contain a valid JSON object.
    Blank lines are skipped. Invalid lines are reported but ignored.

    Args:
        path (Path): Path to the .jsonl file to load.

    Returns:
        Iterable[dict]: An iterable list of parsed JSON objects.
    '''
    samples = []
    with path.open("r", encoding="utf-8") as file:
        # Work across each line
        for idx, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                print(f"[WARN] JSON error in {path.name} line {idx}: {err}")


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
