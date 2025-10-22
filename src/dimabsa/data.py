from pathlib import Path
from typing import List, Dict, Tuple
import json

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def data_dir() -> Path:
    return project_root() / "data"

def load_json(path: Path) -> List[Dict]:
    with path.open() as f:
        return json.load(f)

def load_jsonl(path: Path) -> List[Dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]

class SimpleDataset:
    """
    Replace this with your real fields. This is a stand-in that expects a list of dicts
    with keys: 'x' -> List[float], 'y' -> int.
    """
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple:
        item = self.items[idx]
        return item["x"], item["y"]
