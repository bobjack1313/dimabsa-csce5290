import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.train import parse_args, load_local_dataset

import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension",
    category=UserWarning,
)


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()
    assert args.task == "task1"
    assert args.model == "bert-base-uncased"
    assert args.epochs == 3
    assert args.batch_size == 8
    assert args.lr == 5e-5


def test_parse_args_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--task", "task2", "--epochs", "10"])
    args = parse_args()
    assert args.task == "task2"
    assert args.epochs == 10


def test_load_local_dataset_basic(tmp_path, monkeypatch):
    # Make processed data dir
    base = tmp_path / "data" / "processed" / "task1"
    base.mkdir(parents=True)

    # Create simple valid train.jsonl
    data = '{"Text": "hello", "Quadruplet":[{"Aspect":"food"}]}\n'
    (base / "train.jsonl").write_text(data, encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    ds = load_local_dataset("task1")
    assert "train" in ds
    assert len(ds["train"]) == 1
    assert ds["train"][0]["Aspect"] == ["food"]


def test_load_local_dataset_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_local_dataset("task1")


def test_load_local_dataset_dev_style(tmp_path, monkeypatch):
    base = tmp_path / "data" / "processed" / "task1"
    base.mkdir(parents=True)
    (base / "train.jsonl").write_text(
        '{"Text": "hi", "Aspect": ["staff"]}\n', encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    ds = load_local_dataset("task1")

    rec = ds["train"][0]
    assert rec["Aspect"] == ["staff"]
    assert rec["Quadruplet"][0]["Aspect"] == "staff"
