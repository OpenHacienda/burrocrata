"""Builder: DGT consultas → SFT chat dataset (HF datasets format)."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict

from ..card import BuildResult
from ..formats.sft_chat import to_chat_example
from ..loaders.dgt import iter_consultas

BUILDER_NAME = "dgt-sft"
BUILDER_VERSION = "1"

SYSTEM_PROMPT = (
    "Eres un asistente juridico experto en derecho tributario espanol. "
    "Respondes como la Direccion General de Tributos, citando la normativa aplicable."
)


def build(
    input_dir: Path,
    output_dir: Path,
    val_frac: float = 0.02,
    seed: int = 42,
) -> BuildResult:
    """Build the DGT SFT dataset and persist it to ``output_dir``.

    Filters:
    - skips entries with empty contestacion
    - skips files whose name marks them as anulado (handled in loader)
    """
    rows = iter_consultas(input_dir)

    examples: list[dict] = []
    source_files: list[Path] = []
    skipped_empty_contestacion = 0

    for r in rows:
        if not r.contestacion.strip():
            skipped_empty_contestacion += 1
            continue
        user_content = r.hechos
        if r.cuestion:
            user_content += "\n\n" + r.cuestion
        examples.append(
            to_chat_example(
                system=SYSTEM_PROMPT,
                user=user_content,
                assistant=r.contestacion,
                metadata={
                    "numero": r.numero,
                    "fecha": r.fecha_iso,
                    "normativa": r.normativa,
                    "organo": r.organo,
                },
            )
        )
        source_files.append(r.source_path)

    if not examples:
        raise RuntimeError(f"No usable consultas found under {input_dir}")

    ds = Dataset.from_list(examples)
    splits = ds.train_test_split(test_size=val_frac, seed=seed, shuffle=True)
    dsd = DatasetDict(train=splits["train"], validation=splits["test"])

    output_dir.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(output_dir))

    return BuildResult(
        builder=BUILDER_NAME,
        builder_version=BUILDER_VERSION,
        input_dir=input_dir,
        output_dir=output_dir,
        n_rows=len(examples),
        n_train=len(dsd["train"]),
        n_val=len(dsd["validation"]),
        split_seed=seed,
        val_frac=val_frac,
        source_files=source_files,
        filters=[
            "drop entries with empty contestacion",
            "drop files whose name contains 'anulado'",
        ],
        extra={
            "skipped_empty_contestacion": skipped_empty_contestacion,
            "system_prompt_chars": len(SYSTEM_PROMPT),
        },
    )
