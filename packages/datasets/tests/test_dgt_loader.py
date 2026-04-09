"""Tests for the standalone DGT markdown loader."""

from pathlib import Path

import pytest

from burrocrata.datasets.loaders.dgt import iter_consultas, load_consulta

FIXTURE = Path(__file__).parent / "fixtures" / "data" / "dgt"


def test_iter_skips_anulado_files():
    rows = iter_consultas(FIXTURE)
    numeros = {r.numero for r in rows}
    assert "V0001-99" in numeros
    assert "V0002-99" in numeros
    assert "V0003-99" not in numeros  # anulado in filename
    # V0004-99 has empty contestacion but the loader still returns it;
    # filtering happens at the builder layer.
    assert "V0004-99" in numeros


def test_load_consulta_extracts_sections():
    p = FIXTURE / "consultas" / "2099" / "V0001-99.md"
    row = load_consulta(p)
    assert row is not None
    assert row.numero == "V0001-99"
    assert row.organo == "Test Organo"
    assert row.fecha_iso == "2099-01-01"
    assert "primer caso" in row.hechos
    assert "cuestion" in row.cuestion.lower()
    assert "contestacion" in row.contestacion.lower()


def test_iter_raises_when_no_consultas_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        iter_consultas(tmp_path)
