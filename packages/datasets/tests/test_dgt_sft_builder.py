"""Tests for the dgt-sft builder."""

from pathlib import Path

from datasets import load_from_disk

from burrocrata.datasets.builders.dgt_sft import BUILDER_NAME, build
from burrocrata.datasets.card import write_card

FIXTURE = Path(__file__).parent / "fixtures" / "data" / "dgt"


def test_build_produces_loadable_dataset_and_filters_empty(tmp_path):
    out = tmp_path / "out"
    result = build(input_dir=FIXTURE, output_dir=out, val_frac=0.5, seed=7)

    assert result.builder == BUILDER_NAME
    # 2 valid rows: V0001 and V0002. V0003 anulado, V0004 empty contestacion.
    assert result.n_rows == 2
    assert result.n_train + result.n_val == 2
    assert result.extra["skipped_empty_contestacion"] == 1

    dsd = load_from_disk(str(out))
    assert set(dsd.keys()) == {"train", "validation"}

    sample = dsd["train"][0] if len(dsd["train"]) else dsd["validation"][0]
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert sample["metadata"]["numero"].startswith("V000")


def test_card_writes_with_provenance(tmp_path):
    out = tmp_path / "out"
    result = build(input_dir=FIXTURE, output_dir=out, val_frac=0.5, seed=7)
    card_path = write_card(result)
    assert card_path.exists()
    text = card_path.read_text(encoding="utf-8")
    assert "dgt-sft" in text
    assert "Source fingerprint" in text
    assert "Splits" in text
