"""Native ESMFold2 input conversion and one-seed inference execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from esm.esmfold2_wrapper.compression import read_text_auto
from esm.esmfold2_wrapper.input import PreparedEntity, PreparedInput, load_prepared_input
from esm.esmfold2_wrapper.msa_adapter import (
    native_a3m_to_esmfold2_msa,
    split_a3m_to_esmfold2_msa,
    validate_split_paired_depths,
)

if TYPE_CHECKING:
    from esm.models.esmfold2 import StructurePredictionInput

DEFAULT_CHECKPOINT = "biohub/ESMFold2"


@dataclass(frozen=True)
class InferenceRun:
    """Native inputs and normalized results from one model invocation."""

    prepared_path: Path
    prepared_input: PreparedInput
    structure_input: StructurePredictionInput
    results: tuple[Any, ...]


def _native_id(entity: PreparedEntity) -> str | list[str]:
    return entity.ids[0] if len(entity.ids) == 1 else list(entity.ids)


def _read_declared_msa(content: str | None, path: Path | None) -> str:
    if path is not None:
        if not path.is_absolute():
            raise ValueError(
                "MSA paths must be resolved before native conversion; call "
                "validate_resources(manifest_path) first"
            )
        return read_text_auto(path)
    assert content is not None
    return content


def _protein_msa(entity: PreparedEntity):
    assert entity.kind == "protein" and entity.sequence is not None
    if entity.msa_mode == "none":
        return None
    if entity.msa_mode == "native":
        return native_a3m_to_esmfold2_msa(
            a3m=_read_declared_msa(entity.msa, entity.msa_path),
            query_sequence=entity.sequence,
        )
    return split_a3m_to_esmfold2_msa(
        paired_a3m=_read_declared_msa(entity.paired_msa, entity.paired_msa_path),
        unpaired_a3m=_read_declared_msa(
            entity.unpaired_msa, entity.unpaired_msa_path
        ),
        query_sequence=entity.sequence,
    )


def prepared_to_structure_prediction_input(
    prepared: PreparedInput,
) -> StructurePredictionInput:
    """Convert a resolved manifest to ESMFold2's public SPI dataclasses."""
    validate_split_paired_depths([
        (
            entity.ids[0],
            _read_declared_msa(entity.paired_msa, entity.paired_msa_path),
        )
        for entity in prepared.sequences
        if entity.kind == "protein" and entity.msa_mode == "split"
    ])
    from esm.models.esmfold2 import (
        DNAInput,
        LigandInput,
        Modification,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )

    native_entities = []
    for entity in prepared.sequences:
        entity_id = _native_id(entity)
        if entity.kind == "ligand":
            native_entities.append(
                LigandInput(
                    id=entity_id,
                    smiles=entity.smiles,
                    ccd=None if entity.ccd is None else list(entity.ccd),
                )
            )
            continue

        assert entity.sequence is not None
        modifications = (
            [
                Modification(position=item.position, ccd=item.ccd)
                for item in entity.modifications
            ]
            or None
        )
        if entity.kind == "protein":
            native_entities.append(
                ProteinInput(
                    id=entity_id,
                    sequence=entity.sequence,
                    modifications=modifications,
                    msa=_protein_msa(entity),
                )
            )
        elif entity.kind == "rna":
            native_entities.append(
                RNAInput(
                    id=entity_id,
                    sequence=entity.sequence,
                    modifications=modifications,
                )
            )
        else:
            native_entities.append(
                DNAInput(
                    id=entity_id,
                    sequence=entity.sequence,
                    modifications=modifications,
                )
            )
    return StructurePredictionInput(sequences=native_entities)


def load_structure_prediction_input(
    manifest_path: str | Path,
) -> tuple[PreparedInput, StructurePredictionInput]:
    """Load, resolve and convert one prepared manifest."""
    path = Path(manifest_path).expanduser().resolve()
    prepared = load_prepared_input(path).validate_resources(path)
    return prepared, prepared_to_structure_prediction_input(prepared)


def model_supports_msa(model: Any) -> bool:
    """Return whether a loaded checkpoint consumes full MSA conditioning."""
    config = getattr(model, "config", None)
    msa_config = getattr(config, "msa_encoder", None)
    enabled = bool(getattr(msa_config, "enabled", False))
    if enabled and hasattr(model, "msa_encoder") and model.msa_encoder is None:
        raise RuntimeError(
            "Model config enables the MSA encoder, but the loaded model has none"
        )
    return enabled


def _validate_loaded_model(model: Any) -> Any:
    """Reject native architectures that cannot satisfy the output contract."""
    if getattr(getattr(model, "config", None), "type", None) == "experimental":
        raise ValueError(
            "Experimental ESMFold2 checkpoints are not supported by this wrapper "
            "because they do not produce the required PDE output"
        )
    return model


def _validate_model_options(
    *,
    device: str,
    dtype: str | None,
    esmc_precision: str,
    kernel_backend: str | None,
) -> None:
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if dtype not in {None, "fp32"}:
        raise ValueError("dtype must be null or 'fp32'")
    if esmc_precision not in {"fp32", "bf16", "fp8"}:
        raise ValueError("esmc_precision must be 'fp32', 'bf16', or 'fp8'")
    if kernel_backend not in {None, "fused", "cuequivariance"}:
        raise ValueError(
            "kernel_backend must be null, 'fused', or 'cuequivariance'"
        )


def load_esmfold2_model(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    esmc_checkpoint: str | Path | None = None,
    device: str = "auto",
    dtype: str | None = None,
    esmc_precision: str = "bf16",
    kernel_backend: str | None = None,
    cache_dir: str | Path | None = None,
):
    """Load one public ESMFold2 checkpoint directly onto its inference device."""
    _validate_model_options(
        device=device,
        dtype=dtype,
        esmc_precision=esmc_precision,
        kernel_backend=kernel_backend,
    )
    import torch

    from esm.models.esmfold2 import EsmFold2Model

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {None: None, "fp32": torch.float32}
    kwargs: dict[str, Any] = {
        "device": device,
        "dtype": dtype_map[dtype],
        "esmc_precision": esmc_precision,
        "load_esmc": False,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = os.fspath(cache_dir)
    model = _validate_loaded_model(
        EsmFold2Model.from_pretrained(os.fspath(checkpoint), **kwargs).eval()
    )
    if getattr(model.config, "esmc_config", None) is None:
        model.load_esmc(
            os.fspath(esmc_checkpoint)
            if esmc_checkpoint is not None
            else model.config.esmc_id,
            precision=esmc_precision,
        )
    if kernel_backend is not None:
        model.set_kernel_backend(kernel_backend)
    return model


def _validate_fold_options(
    *,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    seed: int | None,
    lm_dropout: float | None,
    msa_max_depth: int | None,
    msa_column_mask_rate: float,
) -> None:
    for name, value in (
        ("num_loops", num_loops),
        ("num_sampling_steps", num_sampling_steps),
        ("num_diffusion_samples", num_diffusion_samples),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("seed must be null or an integer in the uint32 range")
    if lm_dropout is not None and not 0 <= lm_dropout <= 1:
        raise ValueError("lm_dropout must be null or between 0 and 1")
    if msa_max_depth is not None and (
        type(msa_max_depth) is not int or msa_max_depth < 1
    ):
        raise ValueError("msa_max_depth must be null or a positive integer")
    if not 0 <= msa_column_mask_rate <= 1:
        raise ValueError("msa_column_mask_rate must be between 0 and 1")


def _new_input_builder():
    from esm.models.esmfold2 import ESMFold2InputBuilder

    return ESMFold2InputBuilder()


def _run_with_loaded_model(
    manifest_path: str | Path,
    *,
    prepared: PreparedInput,
    structure_input: StructurePredictionInput,
    model: Any,
    builder: Any | None,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    seed: int | None,
    lm_dropout: float | None,
    msa_max_depth: int | None,
    msa_column_mask_rate: float,
    include_embeddings: bool,
) -> InferenceRun:
    """Run one seed using an already loaded model and input builder."""
    _validate_fold_options(
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
        lm_dropout=lm_dropout,
        msa_max_depth=msa_max_depth,
        msa_column_mask_rate=msa_column_mask_rate,
    )
    has_msa = any(entity.msa_mode != "none" for entity in prepared.sequences)
    if has_msa and not model_supports_msa(model):
        raise ValueError(
            "The selected ESMFold2 checkpoint does not support MSA conditioning; "
            "use a full ESMFold2 checkpoint or remove the MSA inputs"
        )
    if builder is None:
        builder = _new_input_builder()
    raw = builder.fold(
        model,
        structure_input,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
        lm_dropout=lm_dropout,
        msa_max_depth=msa_max_depth,
        msa_column_mask_rate=msa_column_mask_rate,
        include_embeddings=include_embeddings,
        complex_id=prepared.name,
    )
    results = tuple(raw) if isinstance(raw, list) else (raw,)
    if len(results) != num_diffusion_samples:
        raise RuntimeError(
            "ESMFold2 returned a different number of results than requested: "
            f"expected {num_diffusion_samples}, received {len(results)}"
        )
    return InferenceRun(
        prepared_path=Path(manifest_path).expanduser().resolve(),
        prepared_input=prepared,
        structure_input=structure_input,
        results=results,
    )


def run_esmfold2_inference(
    manifest_path: str | Path,
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    esmc_checkpoint: str | Path | None = None,
    device: str = "auto",
    dtype: str | None = None,
    esmc_precision: str = "bf16",
    kernel_backend: str | None = None,
    cache_dir: str | Path | None = None,
    num_loops: int = 20,
    num_sampling_steps: int = 200,
    num_diffusion_samples: int = 5,
    seed: int | None = None,
    lm_dropout: float | None = 0.3,
    msa_max_depth: int | None = 1024,
    msa_column_mask_rate: float = 0.1,
    include_embeddings: bool = False,
) -> InferenceRun:
    """Run one seed through ESMFold2 and normalize its result to a tuple."""
    _validate_model_options(
        device=device,
        dtype=dtype,
        esmc_precision=esmc_precision,
        kernel_backend=kernel_backend,
    )
    _validate_fold_options(
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
        lm_dropout=lm_dropout,
        msa_max_depth=msa_max_depth,
        msa_column_mask_rate=msa_column_mask_rate,
    )
    path = Path(manifest_path).expanduser().resolve()
    prepared, structure_input = load_structure_prediction_input(path)
    model = load_esmfold2_model(
        checkpoint,
        esmc_checkpoint=esmc_checkpoint,
        device=device,
        dtype=dtype,
        esmc_precision=esmc_precision,
        kernel_backend=kernel_backend,
        cache_dir=cache_dir,
    )
    return _run_with_loaded_model(
        path,
        prepared=prepared,
        structure_input=structure_input,
        model=model,
        builder=None,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
        lm_dropout=lm_dropout,
        msa_max_depth=msa_max_depth,
        msa_column_mask_rate=msa_column_mask_rate,
        include_embeddings=include_embeddings,
    )
