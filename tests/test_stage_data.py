import json
from pathlib import Path
from scripts.stage_data import load_jsonl,  write_jsonl
import pytest


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


def test_load_jsonl_handles_varied_schema(tmp_path):
    """Should load records even when schema differs across entries."""
    samples = [
        {
            "ID": "rest26_aspect_va_dev_17",
            "Text": "They make the donuts fresh with each order",
            "Aspect": ["donuts"],
        },
        {
            "ID": "rest26_aspect_va_dev_18",
            "Text": "The staff was very friendly , I will definitely come again",
            "Aspect": ["staff"],
        },
    ]
    path = tmp_path / "varied.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")

    records = load_jsonl(path)
    assert len(records) == 2
    assert all("Text" in r for r in records)
    # Should not raise KeyError even though Quadruplet missing
    assert "Quadruplet" not in records[0] or isinstance(records[0].get("Aspect"), list)


def test_load_jsonl_with_sample_lines(tmp_path: Path):
    '''
    Verify that load_jsonl correctly parses realistic dataset examples.
    '''
    # Arrange: realistic dataset entries
    sample_lines = [
        {
            "ID": "laptop_quad_dev_3",
            "Text": "seems unlikely but whatever , i ' ll go with it .",
            "Quadruplet": [
                {
                    "Aspect": "NULL",
                    "Category": "LAPTOP#GENERAL",
                    "Opinion": "NULL",
                    "VA": "5.00#5.12",
                }
            ],
        },
        {
            "ID": "laptop_quad_dev_4",
            "Text": "this version has been my least favorite version i ' ve had for the following reasons listed bellow the pros .",
            "Quadruplet": [
                {
                    "Aspect": "version",
                    "Category": "LAPTOP#GENERAL",
                    "Opinion": "least favorite",
                    "VA": "3.30#6.60",
                }
            ],
        },
    ]

    file_path = tmp_path / "sample.jsonl"
    file_path.write_text("\n".join(json.dumps(line) for line in sample_lines), encoding="utf-8")

    # Act
    examples = load_jsonl(file_path)

    # Assert
    assert isinstance(examples, list)
    assert len(examples) == 2
    assert examples[0]["ID"] == "laptop_quad_dev_3"
    assert "Quadruplet" in examples[0]
    assert isinstance(examples[0]["Quadruplet"], list)


def test_load_jsonl_with_multiple_quadruplets(tmp_path):
    sample = {
        "ID": "rest16_quad_dev_8",
        "Text": "the food here is rather good , but only if you like to wait for it .",
        "Quadruplet": [
            {"Aspect": "food", "Opinion": "rather good", "Category": "FOOD#QUALITY", "VA": "7.33#7.50"},
            {"Aspect": "NULL", "Opinion": "NULL", "Category": "SERVICE#GENERAL", "VA": "5.00#5.00"},
        ],
    }
    file_path = tmp_path / "multi.jsonl"
    file_path.write_text(json.dumps(sample), encoding="utf-8")

    records = load_jsonl(file_path)
    assert len(records) == 1
    assert len(records[0]["Quadruplet"]) == 2
    assert records[0]["Quadruplet"][0]["Category"] == "FOOD#QUALITY"


def test_load_jsonl_empty_file(tmp_path):
    '''
    Verifies against empty file.
    '''
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("", encoding="utf-8")
    assert load_jsonl(file_path) == []


def test_load_jsonl_skips_bad_lines(tmp_path):
    content = '\n'.join([
        '{"ID": "ok1"}',
        '{bad line}',
        '{"ID": "ok2"}'
    ])
    file_path = tmp_path / "bad.jsonl"
    file_path.write_text(content, encoding="utf-8")
    records = load_jsonl(file_path)
    assert [r["ID"] for r in records] == ["ok1", "ok2"]


def test_load_jsonl_file_not_found():

    with pytest.raises(FileNotFoundError):
        load_jsonl(Path("nonexistent.jsonl"))


def test_load_jsonl_with_unicode(tmp_path):
    sample = {"ID": "unicode_test", "Text": "café — naïve emoji 😊"}
    path = tmp_path / "unicode.jsonl"
    path.write_text(json.dumps(sample), encoding="utf-8")
    result = load_jsonl(path)
    assert result[0]["Text"].startswith("café")


def test_write_jsonl_creates_file_and_writes(tmp_path: Path):
    """Should create the file and correctly write all JSON objects."""
    samples = [
        {"ID": "1", "Text": "First sample"},
        {"ID": "2", "Text": "Second sample"},
    ]
    out_file = tmp_path / "out" / "samples.jsonl"

    # Act
    write_jsonl(samples, out_file)

    # Assert: file should exist and contain two valid JSON lines
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed == samples


def test_write_jsonl_overwrites_existing(tmp_path: Path):
    """Should overwrite existing files, not append."""
    path = tmp_path / "data.jsonl"
    path.write_text('{"old": "data"}\n', encoding="utf-8")

    samples = [{"id": "new"}]
    write_jsonl(samples, path)

    content = path.read_text(encoding="utf-8").strip()
    assert content == json.dumps(samples[0], ensure_ascii=False)


def test_write_jsonl_creates_parent_dirs(tmp_path: Path):
    """Should automatically create parent directories if missing."""
    nested_path = tmp_path / "deep" / "nested" / "output.jsonl"
    samples = [{"msg": "hello"}]

    write_jsonl(samples, nested_path)

    assert nested_path.exists()
    data = json.loads(nested_path.read_text(encoding="utf-8").strip())
    assert data["msg"] == "hello"


def test_write_jsonl_handles_unicode(tmp_path: Path):
    """Should correctly handle UTF-8 and special characters."""
    samples = [{"text": "café — naïve emoji 😊"}]
    path = tmp_path / "unicode.jsonl"

    write_jsonl(samples, path)

    text = path.read_text(encoding="utf-8")
    assert "😊" in text
    obj = json.loads(text)
    assert obj["text"].startswith("café")

