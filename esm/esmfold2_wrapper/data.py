"""Materialize portable MSA resources for an ESMFold2 prepared bundle."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from esm.esmfold2_wrapper.compression import read_text_auto, write_zstd_text
from esm.esmfold2_wrapper.input import (
    PreparedEntity,
    PreparedInput,
    load_prepared_input,
    write_prepared_input,
)
from esm.esmfold2_wrapper.msa_adapter import (
    split_a3m_to_keyed_text,
    validate_native_a3m,
    validate_split_paired_depths,
)


@dataclass(frozen=True)
class _MSAResources:
    mode: str
    native: str | None = None
    paired: str | None = None
    unpaired: str | None = None


def _canonical_text(text: str) -> str:
    return text if not text or text.endswith("\n") else f"{text}\n"


def _read_source(content: str | None, path: Path | None) -> str | None:
    return read_text_auto(path) if path is not None else content


def _relative_path(resource: Path, manifest: Path) -> Path:
    return Path(os.path.relpath(resource, manifest.parent))


def _resource_path(msa_dir: Path, filename: str, manifest: Path) -> Path:
    """Resolve a resource destination without following an escaping msas symlink."""
    job_dir = manifest.parent.resolve()
    resource = (msa_dir / filename).resolve()
    if job_dir not in resource.parents:
        raise ValueError(
            f"Prepared MSA path escapes the job directory: {resource}"
        )
    return resource


def _load_and_validate_msa(entity: PreparedEntity) -> _MSAResources:
    assert entity.kind == "protein"
    assert entity.sequence is not None
    if entity.msa_mode == "none":
        return _MSAResources(mode="none")
    if entity.msa_mode == "native":
        native = _read_source(entity.msa, entity.msa_path)
        assert native is not None
        validate_native_a3m(a3m=native, query_sequence=entity.sequence)
        return _MSAResources(mode="native", native=_canonical_text(native))

    paired = _read_source(entity.paired_msa, entity.paired_msa_path)
    unpaired = _read_source(entity.unpaired_msa, entity.unpaired_msa_path)
    assert paired is not None and unpaired is not None
    split_a3m_to_keyed_text(
        paired_a3m=paired,
        unpaired_a3m=unpaired,
        query_sequence=entity.sequence,
    )
    return _MSAResources(
        mode="split",
        paired=_canonical_text(paired),
        unpaired=_canonical_text(unpaired),
    )


def _materialize_entity(
    entity: PreparedEntity,
    resource: _MSAResources,
    *,
    target_name: str,
    msa_dir: Path,
    manifest: Path,
    compress_fold_input: bool,
) -> PreparedEntity:
    if resource.mode == "none":
        return entity
    stem = f"{target_name}__{entity.ids[0]}"
    suffix = ".zst" if compress_fold_input else ""
    if resource.mode == "native":
        assert resource.native is not None
        destination = _resource_path(msa_dir, f"{stem}_msa.a3m{suffix}", manifest)
        path = write_zstd_text(destination, resource.native, compress=compress_fold_input)
        return replace(
            entity,
            msa=None,
            msa_path=_relative_path(path, manifest),
        )

    assert resource.paired is not None and resource.unpaired is not None
    paired_path = write_zstd_text(
        _resource_path(msa_dir, f"{stem}_pairedmsa.a3m{suffix}", manifest),
        resource.paired,
        compress=compress_fold_input,
    )
    unpaired_path = write_zstd_text(
        _resource_path(msa_dir, f"{stem}_unpairedmsa.a3m{suffix}", manifest),
        resource.unpaired,
        compress=compress_fold_input,
    )
    return replace(
        entity,
        paired_msa=None,
        paired_msa_path=_relative_path(paired_path, manifest),
        unpaired_msa=None,
        unpaired_msa_path=_relative_path(unpaired_path, manifest),
    )


def _load_validated_sources(
    input_path: str | Path,
) -> tuple[PreparedInput, list[PreparedEntity], list[_MSAResources]]:
    """Read and validate existing conditions without creating public files."""
    source_manifest = Path(input_path).expanduser().resolve()
    prepared = load_prepared_input(source_manifest)
    resolved = prepared.validate_resources(source_manifest)

    protein_entities = [
        entity for entity in resolved.sequences if entity.kind == "protein"
    ]
    protein_resources = [_load_and_validate_msa(entity) for entity in protein_entities]
    validate_split_paired_depths([
        (entity.ids[0], resource.paired)
        for entity, resource in zip(protein_entities, protein_resources, strict=True)
        if resource.mode == "split" and resource.paired is not None
    ])
    return prepared, protein_entities, protein_resources


def validate_data_input(input_path: str | Path) -> None:
    """Run the lightweight data pipeline without publishing a snapshot."""
    _load_validated_sources(input_path)


def prepare_data_bundle(
    input_path: str | Path, output_manifest_path: str | Path,
    *, compress_fold_input: bool = False,
) -> PreparedInput:
    """Validate and publish portable MSA resources plus the final manifest.

    All source MSAs are parsed before anything is written. Each resource is
    replaced atomically and the manifest is published last. If rewriting an
    existing fixed-name bundle is interrupted, rerun with write_input_json=True
    using the original source input to refresh it.
    """
    output_manifest = Path(output_manifest_path).expanduser().resolve()
    prepared, protein_entities, protein_resources = _load_validated_sources(input_path)

    suffix = ".zst" if compress_fold_input else ""
    resource_names: list[str] = []
    for entity, resource in zip(protein_entities, protein_resources, strict=True):
        stem = f"{prepared.name}__{entity.ids[0]}"
        if resource.mode == "native":
            resource_names.append(f"{stem}_msa.a3m{suffix}")
        elif resource.mode == "split":
            resource_names.extend(
                [f"{stem}_pairedmsa.a3m{suffix}", f"{stem}_unpairedmsa.a3m{suffix}"]
            )
    folded_names = [name.casefold() for name in resource_names]
    if len(folded_names) != len(set(folded_names)):
        raise ValueError(
            "Prepared MSA resource names collide on a case-insensitive filesystem"
        )

    msa_dir = output_manifest.parent / "msas"
    rewritten_proteins = iter(
        _materialize_entity(
            entity,
            resource,
            target_name=prepared.name,
            msa_dir=msa_dir,
            manifest=output_manifest,
            compress_fold_input=compress_fold_input,
        )
        for entity, resource in zip(protein_entities, protein_resources, strict=True)
    )
    rewritten = tuple(
        next(rewritten_proteins) if entity.kind == "protein" else entity
        for entity in prepared.sequences
    )
    bundled = replace(prepared, sequences=rewritten)
    write_prepared_input(bundled, output_manifest)
    return bundled
