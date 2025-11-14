import shutil
from scripts.stage_data import make_dir, copy_data_to_raw


def test_make_dir_creates_directory(tmp_path):
    target = tmp_path / "nested" / "dir"
    make_dir(target)
    assert target.exists()
    assert target.is_dir()


def test_copy_data_to_raw_copies_file(tmp_path, monkeypatch):
    src = tmp_path / "src.jsonl"
    dst = tmp_path / "dst.jsonl"
    src.write_text("line1\n")

    copy_data_to_raw(src, dst)
    assert dst.exists()
    assert dst.read_text() == "line1\n"


def test_copy_data_handles_missing_file(tmp_path, capsys):
    src = tmp_path / "missing.jsonl"
    dst = tmp_path / "unused.jsonl"

    copy_data_to_raw(src, dst)
    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert not dst.exists()
