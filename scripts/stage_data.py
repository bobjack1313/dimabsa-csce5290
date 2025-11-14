#!/usr/bin/env python3
# =============================================================================
# Script Name : stage_data.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
#
# Description : Stage raw DimABSA 2026 dataset files into the local project
#               structure. This script copies the official English subtask
#               files from the external DimABSA2026 repository into the
#               project's data directory under data/raw/.
#
# Exp inputs  : external/DimABSA2026/task-dataset/track_a/subtask_1/eng/
#               external/DimABSA2026/task-dataset/track_a/subtask_2/eng/
#
# Exp outputs : data/raw/task1/*.jsonl
#               data/raw/task2/*.jsonl
#
# Usage       : python stage_data.py
#
# Notes       : - This script does NOT download any data. The repository
#                 external/DimABSA2026 must already be cloned.
#               - This only stages raw files; preprocessing happens in
#                 prepare_task1.py and prepare_task2.py.
#               - Safe to rerun. Raw files will be overwritten cleanly.
#
# Author      : Amrit Adhikari, Bob Jack - Group 4
# Date        : 2025-10-29
# =============================================================================
from __future__ import annotations
from pathlib import Path
import shutil


# File Paths
ROOT = Path(__file__).resolve().parent.parent
SRC_BASE = ROOT / "external" / "DimABSA2026" / "task-dataset" / "track_a"
OUT_BASE_RAW = ROOT / "data" / "raw"
OUT_BASE_RAW_TASK1 = OUT_BASE_RAW / "task1"
OUT_BASE_RAW_TASK2 = OUT_BASE_RAW / "task2"


# ---- Helper Functions -----
def make_dir(path: Path) -> None:
    '''
    Create the directory if it does not exist.

    Parameters:
        path : Path - Directory path to create.
    '''
    path.mkdir(parents=True, exist_ok=True)


def copy_data_to_raw(source_file: Path, dest_file: Path) -> None:
    '''
    Copy a single file from external/ to data/raw/.

    Parameters:

        source_file : Path - Path to source JSONL file.
        dest_file : Path - Destination file inside data/raw/.

    Notes:
        - If the source file is missing, prints a warning instead of failing.
    '''
    if not source_file.exists():
        print(f"[WARN] Missing expected file: {source_file}")
        return

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_file, dest_file)

    print(f"Copied - {dest_file.name}")


# ---- Staging -----
def stage_raw_data_task1() -> None:
    '''
    Stage raw DimABSA Task 1 input files into data/raw/task1/.
    '''
    print("\nStaging Task 1 raw files...")

    src = SRC_BASE / "subtask_1" / "eng"

    train_files = [
        "eng_laptop_train_alltasks.jsonl",
        "eng_restaurant_train_alltasks.jsonl",
    ]
    valid_files = [
        "eng_laptop_dev_task1.jsonl",
        "eng_restaurant_dev_task1.jsonl",
    ]

    for file_name in train_files:
        copy_data_to_raw(src / file_name, OUT_BASE_RAW_TASK1 / file_name)

    for file_name in valid_files:
        copy_data_to_raw(src / file_name, OUT_BASE_RAW_TASK1 / file_name)


def stage_raw_data_task2() -> None:
    '''
    Stage raw DimABSA Task 2 input files into data/raw/task2/.
    '''
    print("\nStaging Task 2 raw files...")

    src = SRC_BASE / "subtask_2" / "eng"

    train_files = [
        "eng_laptop_train_alltasks.jsonl",
        "eng_restaurant_train_alltasks.jsonl",
    ]
    valid_files = [
        "eng_laptop_dev_task2.jsonl",
        "eng_restaurant_dev_task2.jsonl",
    ]

    for file_name in train_files:
        copy_data_to_raw(src / file_name, OUT_BASE_RAW_TASK2 / file_name)

    for file_name in valid_files:
        copy_data_to_raw(src / file_name, OUT_BASE_RAW_TASK2 / file_name)


# ---- Main -----
def main() -> None:
    print("Preparing DimABSA 2026 English raw datasets...")
    stage_raw_data_task1()
    stage_raw_data_task2()
    print("All done!")


# Entry point
if __name__ == "__main__":
    main()
