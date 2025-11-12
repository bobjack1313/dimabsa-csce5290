#!/usr/bin/env python
"""
Prepare DimABSA2026 English data (Task 1 = DimASR, Task 2 = DimASTE)
into data/processed/. Works entirely from the local submodule:
external/DimABSA2026/task-dataset/track_a/subtask_{1,2}/eng
"""
from __future__ import annotations
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_BASE = ROOT / "external" / "DimABSA2026" / "task-dataset" / "track_a"
OUT_BASE = ROOT / "data" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] bad JSON line in {path.name}: {e}")
    return items


def write_jsonl(items: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {out}  ({len(items)} records)")


def process_task1() -> None:
    """Task 1 – DimASR (Aspect Sentiment Regression)"""
    src = SRC_BASE / "subtask_1" / "eng"
    train_files = [
        src / "eng_restaurant_train_alltasks.jsonl",
        src / "eng_laptop_train_alltasks.jsonl",
    ]
    valid_files = [
        src / "eng_restaurant_dev_task1.jsonl",
        src / "eng_laptop_dev_task1.jsonl",
    ]

    train, valid = [], []
    for p in train_files:
        if p.exists():
            train += load_jsonl(p)
        else:
            print(f"[WARN] missing {p}")
    for p in valid_files:
        if p.exists():
            valid += load_jsonl(p)
        else:
            print(f"[WARN] missing {p}")

    # sanity-check fields
    def check_fields(items: list[dict]) -> None:
        missing = [x["ID"] for x in items if "Text" not in x or "Aspect" not in x]
        if missing:
            print(f"[WARN] {len(missing)} records missing Text/Aspect fields")

    check_fields(train)
    check_fields(valid)

    write_jsonl(train, OUT_BASE / "task1_train.jsonl")
    write_jsonl(valid, OUT_BASE / "task1_valid.jsonl")


def process_task2() -> None:
    """Task 2 – DimASTE (Aspect–Opinion–VA Triplet Extraction)"""
    src = SRC_BASE / "subtask_2" / "eng"
    train_files = [
        src / "eng_restaurant_train_alltasks.jsonl",
        src / "eng_laptop_train_alltasks.jsonl",
    ]
    valid_files = [
        src / "eng_restaurant_dev_task2.jsonl",
        src / "eng_laptop_dev_task2.jsonl",
    ]

    train, valid = [], []
    for p in train_files:
        if p.exists():
            train += load_jsonl(p)
        else:
            print(f"[WARN] missing {p}")
    for p in valid_files:
        if p.exists():
            valid += load_jsonl(p)
        else:
            print(f"[WARN] missing {p}")

    def check_fields(items: list[dict]) -> None:
        missing = [x["ID"] for x in items if "Text" not in x]
        if missing:
            print(f"[WARN] {len(missing)} records missing Text")

    check_fields(train)
    check_fields(valid)

    write_jsonl(train, OUT_BASE / "task2_train.jsonl")
    write_jsonl(valid, OUT_BASE / "task2_valid.jsonl")



def main() -> None:
    print("Preparing DimABSA 2026 English datasets…")
    process_task1()
    process_task2()
    print("All done!")


# Entry Point
if __name__ == "__main__":
    main()

