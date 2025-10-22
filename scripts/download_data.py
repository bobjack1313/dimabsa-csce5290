#!/usr/bin/env python
from __future__ import annotations
import argparse, json, os, sys, zipfile, hashlib
from pathlib import Path
from typing import Iterable, Dict, Tuple

try:
    import requests
except ImportError:
    requests = None

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def download_file(url: str, out: Path) -> Path:
    if requests is None:
        raise RuntimeError("requests not installed. pip install requests")
    out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with out.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 18):
            if chunk:
                f.write(chunk)
    return out

def write_jsonl(items: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def from_hf(dataset_name: str, subset: str | None, split: str) -> Iterable[Dict]:
    """Convert a HuggingFace datasets split into a generic JSONL format.
       Adjust field extraction here for your ABSA dataset."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
    d = ds[split]
    # EXAMPLE MAPPING (adjust to your dataset):
    # Expect each row to have fields: text, label (int), aspect (optional)
    for row in d:
        text = row.get("text") or row.get("sentence") or row.get("review") or ""
        label = row.get("label") if "label" in row else row.get("polarity")
        aspect = row.get("aspect") or row.get("target")  # may be None
        yield {"text": text, "label": label, "aspect": aspect}

def from_url_zip(url: str, raw_dir: Path) -> Path:
    """Download a ZIP and extract under data/raw/"""
    zip_path = raw_dir / "dataset.zip"
    print(f"Downloading: {url}")
    download_file(url, zip_path)
    print("Saved:", zip_path, "sha256:", sha256_of(zip_path)[:12])
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)
    print("Extracted to:", raw_dir)
    return raw_dir

def main():
    ap = argparse.ArgumentParser(description="Download and prepare dataset into data/raw and data/processed")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf", type=str, help="HuggingFace dataset name, e.g., sem_eval_2016_task5")
    ap.add_argument("--subset", type=str, default=None, help="HF subset/config name (if any)")
    g.add_argument("--url", type=str, help="Direct URL to a ZIP or JSON/JSONL")

    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--valid-split", type=str, default="validation")
    ap.add_argument("--test-split",  type=str, default="test")
    ap.add_argument("--out-root",    type=Path, default=Path("data"))
    ap.add_argument("--skip-test",   action="store_true")
    args = ap.parse_args()

    raw_dir = ensure_dir(args.out_root / "raw")
    proc_dir = ensure_dir(args.out_root / "processed")

    if args.hf:
        # HuggingFace path
        print(f"Loading from HF: {args.hf} subset={args.subset!r}")
        splits = [(args.train_split, "train.jsonl"), (args.valid_split, "valid.jsonl")]
        if not args.skip_test:
            splits.append((args.test_split, "test.jsonl"))
        for split_name, fname in splits:
            try:
                rows = list(from_hf(args.hf, args.subset, split_name))
            except Exception as e:
                print(f"[WARN] Could not load split {split_name}: {e}")
                if split_name in ("validation", "valid", "dev"):
                    print("Skipping valid split; you can create one later via script.")
                continue
            out = proc_dir / fname
            write_jsonl(rows, out)
            print(f"Wrote {out}   count={len(rows)}")

    else:
        # Direct URL path
        print("Downloading from URL…")
        target = args.url.lower()
        if target.endswith(".zip"):
            from_url_zip(args.url, raw_dir)
            print("NOTE: Map the extracted files to JSONL in data/processed next (custom).")
        elif target.endswith(".json") or target.endswith(".jsonl"):
            # Single file download
            name = "download.jsonl" if target.endswith(".jsonl") else "download.json"
            out = raw_dir / name
            download_file(args.url, out)
            print("Saved:", out, "sha256:", sha256_of(out)[:12])
            print("NOTE: Convert raw JSON to JSONL under data/processed/ with your schema.")
        else:
            print("Unsupported URL type. Provide a .zip, .json, or .jsonl")

    print("Done.")

if __name__ == "__main__":
    sys.exit(main())

