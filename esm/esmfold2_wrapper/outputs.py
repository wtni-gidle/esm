"""Publish ESMFold2 results in the shared seed/sample prediction layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class PredictionOutputError(ValueError):
    """Raised when native results cannot satisfy the output contract."""


@dataclass(frozen=True)
class PublishedSample:
    """Canonical output paths for one seed/sample pair."""

    seed: int
    sample: int
    model_path: Path
    summary_path: Path
    plddt_path: Path
    pae_path: Path
    pde_path: Path


@dataclass(frozen=True)
class _Artifact:
    path: Path
    kind: str
    value: Any


def _validate_seed_and_count(seed: int, sample_count: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise PredictionOutputError("seed must be an integer in the uint32 range")
    if type(sample_count) is not int or sample_count < 1:
        raise PredictionOutputError("sample_count must be a positive integer")


def expected_seed_samples(
    predictions_dir: str | Path,
    *,
    seed: int,
    sample_count: int,
) -> tuple[PublishedSample, ...]:
    """Return the exact canonical paths expected for one seed."""
    _validate_seed_and_count(seed, sample_count)
    root = Path(predictions_dir).expanduser().resolve()
    return tuple(
        PublishedSample(
            seed=seed,
            sample=sample,
            model_path=root / "models" / f"seed-{seed}_sample-{sample}_model.cif",
            summary_path=(
                root
                / "summary_confidences"
                / f"seed-{seed}_sample-{sample}_summary_confidences.json"
            ),
            plddt_path=(
                root / "full_data" / f"plddt_seed-{seed}_sample-{sample}.npz"
            ),
            pae_path=root / "full_data" / f"pae_seed-{seed}_sample-{sample}.npz",
            pde_path=root / "full_data" / f"pde_seed-{seed}_sample-{sample}.npz",
        )
        for sample in range(sample_count)
    )


def expected_embedding_path(predictions_dir: str | Path, *, seed: int) -> Path:
    """Return the seed-level embedding path."""
    _validate_seed_and_count(seed, 1)
    root = Path(predictions_dir).expanduser().resolve()
    return root / "embeddings" / f"seed-{seed}_embeddings.npz"


def seed_outputs_complete(
    predictions_dir: str | Path,
    *,
    seed: int,
    sample_count: int,
    include_embeddings: bool = False,
) -> bool:
    """Return whether every canonical artifact for one seed exists as a file."""
    try:
        expected = expected_seed_samples(
            predictions_dir, seed=seed, sample_count=sample_count
        )
        for sample in expected:
            if not all(
                path.is_file()
                for path in (
                    sample.model_path,
                    sample.summary_path,
                    sample.plddt_path,
                    sample.pae_path,
                    sample.pde_path,
                )
            ):
                return False
        if include_embeddings and not expected_embedding_path(
            predictions_dir, seed=seed
        ).is_file():
            return False
        return True
    except (OSError, PredictionOutputError):
        return False


def _numeric_array(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise PredictionOutputError(f"Native result is missing required {name}")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float") and "bfloat16" in str(getattr(value, "dtype", "")):
        value = value.float()
    if hasattr(value, "contiguous"):
        value = value.contiguous()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        raise PredictionOutputError(f"{name} must be a non-empty numeric array")
    if not np.isfinite(array).all():
        raise PredictionOutputError(f"{name} contains non-finite values")
    return array


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        raise PredictionOutputError(f"{name} must be finite")
    return scalar


def _chain_pair_summary(result: Any) -> tuple[list[str], dict[str, dict[str, float]]]:
    lookup = getattr(result.complex.metadata, "chain_lookup", {})
    chain_ids = [str(lookup[index]) for index in sorted(lookup)]
    matrix = _numeric_array(result.pair_chains_iptm, "pair_chains_iptm")
    if matrix.shape != (len(chain_ids), len(chain_ids)):
        raise PredictionOutputError(
            "pair_chains_iptm shape does not match the serialized chain count"
        )
    return chain_ids, {
        first: {
            second: float(matrix[first_index, second_index])
            for second_index, second in enumerate(chain_ids)
        }
        for first_index, first in enumerate(chain_ids)
    }


def _sample_artifacts(result: Any, paths: PublishedSample) -> list[_Artifact]:
    complex_object = getattr(result, "complex", None)
    if complex_object is None or not callable(getattr(complex_object, "to_mmcif", None)):
        raise PredictionOutputError("Native result is missing a serializable complex")
    cif_text = complex_object.to_mmcif()
    if not isinstance(cif_text, str) or not cif_text.strip():
        raise PredictionOutputError("Native result produced an empty mmCIF")

    plddt = _numeric_array(getattr(result, "plddt", None), "plddt")
    structure_plddt = _numeric_array(
        getattr(complex_object, "plddt", None), "structure_token_plddt"
    )
    pae = _numeric_array(getattr(result, "pae", None), "pae")
    pde = _numeric_array(getattr(result, "pde", None), "pde")
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise PredictionOutputError("pae must be a square model-token matrix")
    if pde.shape != pae.shape:
        raise PredictionOutputError("pde and pae must have the same shape")
    if plddt.ndim != 1 or plddt.shape[0] != pae.shape[0]:
        raise PredictionOutputError("plddt and pae model-token axes do not match")

    chain_ids, pair_chains = _chain_pair_summary(result)
    summary = {
        "seed": paths.seed,
        "sample": paths.sample,
        "mean_plddt": float(structure_plddt.mean()),
        "ptm": _optional_float(getattr(result, "ptm", None), "ptm"),
        "iptm": _optional_float(getattr(result, "iptm", None), "iptm"),
        "chain_ids": chain_ids,
        "pair_chains_iptm": pair_chains,
    }
    return [
        _Artifact(paths.model_path, "text", cif_text),
        _Artifact(
            paths.plddt_path,
            "npz",
            {"plddt": plddt, "structure_token_plddt": structure_plddt},
        ),
        _Artifact(paths.pae_path, "npz", {"pae": pae}),
        _Artifact(paths.pde_path, "npz", {"pde": pde}),
        _Artifact(paths.summary_path, "json", summary),
    ]


def _embedding_artifact(result: Any, path: Path) -> _Artifact:
    values = {
        "pair_pooled": _numeric_array(
            getattr(result, "output_embedding_pair_pooled", None),
            "output_embedding_pair_pooled",
        )
    }
    for field in ("output_embedding_sequence", "residue_index", "entity_id"):
        value = getattr(result, field, None)
        if value is not None:
            values[field] = _numeric_array(value, field)
    return _Artifact(path, "npz", values)


def _write_artifact(artifact: _Artifact) -> None:
    artifact.path.parent.mkdir(parents=True, exist_ok=True)
    if artifact.kind == "text":
        artifact.path.write_text(artifact.value, encoding="utf-8")
    elif artifact.kind == "json":
        artifact.path.write_text(
            json.dumps(artifact.value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        np.savez_compressed(artifact.path, **artifact.value)


def publish_inference_results(
    results: tuple[Any, ...] | list[Any],
    *,
    predictions_dir: str | Path,
    seed: int,
    include_embeddings: bool = False,
) -> tuple[PublishedSample, ...]:
    """Validate and write every sample from one seed."""
    normalized = tuple(results)
    expected = expected_seed_samples(
        predictions_dir, seed=seed, sample_count=len(normalized)
    )
    artifacts = [
        artifact
        for result, paths in zip(normalized, expected, strict=True)
        for artifact in _sample_artifacts(result, paths)
    ]
    if include_embeddings:
        artifacts.append(
            _embedding_artifact(
                normalized[0], expected_embedding_path(predictions_dir, seed=seed)
            )
        )

    # Match the output convention by writing summaries after model/full-data files.
    artifacts.sort(key=lambda artifact: artifact.kind == "json")
    for artifact in artifacts:
        _write_artifact(artifact)
    return expected
