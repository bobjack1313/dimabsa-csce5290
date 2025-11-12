import json
from pathlib import Path
from scripts.stage_data import load_jsonl


def test_load_jsonl_reads_valid_lines(tmp_path: Path):
    """Should correctly parse valid JSON lines and skip invalid or blank ones."""
    # Arrange: build a temporary .jsonl file
    sample_lines = [
        json.dumps({"id": 1, "text": "batches"}),
        "",  # blank line
        "{bad json}",  # malformed
        json.dumps({"id": 2, "text": "many"})
    ]
    file_path = tmp_path / "sample.jsonl"
    file_path.write_text("\n".join(sample_lines), encoding="utf-8")

    # Act
    examples = load_jsonl(file_path)

    # Assert
    assert isinstance(examples, list)
    assert examples == [
        {"id": 1, "text": "batches"},
        {"id": 2, "text": "many"}
    ]

