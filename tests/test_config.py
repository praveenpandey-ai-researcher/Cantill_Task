"""Config loader tests. These use tiny temp files, never the real task data."""
from pathlib import Path

import pytest

from spendpipe.config import ConfigError, load_config


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


def test_minimal_config_loads_and_merges_defaults(tmp_path):
    _write(tmp_path, "lines.csv", "id,amt\n1,10\n")
    _write(tmp_path, "tree.csv", "l1,l2\nA,B\n")
    cfg_path = _write(tmp_path, "c.yaml", """
company: T
data_dir: .
taxonomy: {file: tree.csv}
sources:
  - name: S
    file: lines.csv
    columns: {record_id: id, amount: amt}
""")
    cfg = load_config(cfg_path)
    assert cfg.company == "T"
    assert cfg.reporting_currency == "USD"            # default
    assert cfg.sources[0]["role"] == "primary"        # source default
    assert cfg.file("lines.csv").exists()


def test_missing_required_column_and_unknown_field_rejected(tmp_path):
    _write(tmp_path, "lines.csv", "id\n1\n")
    _write(tmp_path, "tree.csv", "l1\nA\n")
    cfg_path = _write(tmp_path, "c.yaml", """
taxonomy: {file: tree.csv}
sources:
  - name: S
    file: lines.csv
    columns: {record_id: id, colour: id}
""")
    with pytest.raises(ConfigError) as err:
        load_config(cfg_path)
    msg = str(err.value)
    assert "columns.amount" in msg and "colour" in msg


def test_data_dir_override(tmp_path):
    other = tmp_path / "blind"
    other.mkdir()
    _write(other, "lines.csv", "id,amt\n1,10\n")
    _write(other, "tree.csv", "l1\nA\n")
    cfg_path = _write(tmp_path, "c.yaml", """
data_dir: does_not_exist
taxonomy: {file: tree.csv}
sources:
  - {name: S, file: lines.csv, columns: {record_id: id, amount: amt}}
""")
    cfg = load_config(cfg_path, data_dir=other)
    assert cfg.data_dir == other.resolve()


def test_reference_files_fall_back_but_sources_do_not(tmp_path):
    home = tmp_path / "home"
    blind = tmp_path / "blind"
    home.mkdir()
    blind.mkdir()
    _write(home, "tree.csv", "l1\nA\n")              # tree only in the config's own folder
    _write(home, "lines.csv", "id,amt\n1,10\n")
    _write(blind, "lines.csv", "id,amt\n2,20\n")
    cfg_path = _write(tmp_path, "c.yaml", """
data_dir: home
taxonomy: {file: tree.csv}
sources:
  - {name: S, file: lines.csv, columns: {record_id: id, amount: amt}}
""")
    cfg = load_config(cfg_path, data_dir=blind)
    assert cfg.file("tree.csv", fallback=True) == home / "tree.csv"
    assert cfg.file("lines.csv") == blind / "lines.csv"      # new data, never the old file
    (blind / "lines.csv").unlink()
    with pytest.raises(ConfigError):
        load_config(cfg_path, data_dir=blind)
