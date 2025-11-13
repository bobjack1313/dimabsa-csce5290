#!/usr/bin/env python3
# =============================================================================
# Script Name : stage_data.py
# Project     : DimABSA 2026 (CSCE 5290 Term Project)
# Description : Stage raw DimABSA 2026 dataset files into the local data
#               pipeline. This script copies the English subtask files from
#               the external DimABSA2026 repository into the project's data
#               directory. Preprocessing prep is performed for the two tasks.
#
# Exp inputs  : external/DimABSA2026/task-dataset/track_a/subtask_1/eng/
#               external/DimABSA2026/task-dataset/track_a/subtask_2/eng/
#
# Exp outputs : data/processed/task1/train.jsonl
#               data/processed/task1/valid.jsonl
#               data/processed/task2/train.jsonl
#               data/processed/task2/valid.jsonl
#
# Usage       : python stage_data.py
#
# Notes       : - This script does not download from the internet. Must
#                 have sub repo under external/
#               - It verifies source file presence and checks for naming.
#               - Safe to rerun. Existing files are overwritten with new.
#
# Author      : Amrit Adhikari, Bob Jack - Group 4
# Date        : 2025-10-29
# =============================================================================
from __future__ import annotations
import json
from pathlib import Path

# File Paths
ROOT = Path(__file__).resolve().parent.parent
SRC_BASE = ROOT / "external" / "DimABSA2026" / "task-dataset" / "track_a"
OUT_BASE = ROOT / "data" / "processed"



