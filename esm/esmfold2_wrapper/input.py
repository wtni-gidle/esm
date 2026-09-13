"""Versioned JSON contract for the EnsembleFold ESMFold2 wrapper.

This module deliberately lives outside :mod:`esm.models.esmfold2` so data-only
work does not import Torch or initialize any model-side resources.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

PREPARED_INPUT_VERSION = 1

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_TOP_LEVEL_REQUIRED_FIELDS = {"version", "name", "sequences"}
_TOP_LEVEL_OPTIONAL_FIELDS: set[str] = set()
_ENTITY_TYPES = {"protein", "rna", "dna", "ligand"}
_COMMON_POLYMER_FIELDS = {"type", "id", "sequence", "modifications"}
_ENTITY_FIELDS = {
    "protein": _COMMON_POLYMER_FIELDS
    | {
        "msa",
        "msaPath",
        "pairedMsa",
        "pairedMsaPath",
        "unpairedMsa",
        "unpairedMsaPath",
    },
    "rna": _COMMON_POLYMER_FIELDS,
    "dna": _COMMON_POLYMER_FIELDS,
    "ligand": {"type", "id", "smiles", "ccd"},
}

MSAMode = Literal["none", "native", "split"]


class PreparedInputError(ValueError):
    """Raised when wrapper input does not satisfy the public contract."""


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparedInputError(f"{location} must be a JSON object")
    return value


def _expect_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    location: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        raise PreparedInputError(
            f"{location} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise PreparedInputError(
            f"{location} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _expect_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PreparedInputError(f"{location} must be a non-empty string")
    return value


def _optional_string(
    value: Any, location: str, *, allow_empty: bool = False
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value or (not value and not allow_empty):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise PreparedInputError(f"{location} must be {qualifier} or null")
    return value


def _optional_path(value: Any, location: str) -> Path | None:
    if value is None:
        return None
    return Path(_expect_nonempty_string(value, location))


def _validate_safe_name(value: Any, location: str) -> str:
    value = _expect_nonempty_string(value, location)
    if not _SAFE_NAME.fullmatch(value):
        raise PreparedInputError(
            f"{location} must contain only ASCII letters, digits, '_', '-', and '.', "
            "and must start with a letter, digit, or '_'"
        )
    return value


def _parse_ids(value: Any, location: str) -> tuple[str, ...]:
    raw_ids = value if isinstance(value, list) else [value]
    if not raw_ids:
        raise PreparedInputError(f"{location} must not be an empty list")
    ids = tuple(
        _validate_safe_name(item, f"{location}[{index}]")
        for index, item in enumerate(raw_ids)
    )
    if len(ids) != len(set(ids)):
        raise PreparedInputError(f"{location} contains duplicate chain IDs")
    return ids


def _resolve_path(path: Path, manifest_path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (manifest_path.expanduser().resolve().parent / expanded).resolve()


@dataclass(frozen=True)
class PreparedModification:
    """A zero-indexed polymer modification using an ESMFold2 CCD code."""

    position: int
    ccd: str

    @classmethod
    def from_dict(cls, value: Any, location: str) -> PreparedModification:
        data = _expect_mapping(value, location)
        _expect_fields(
            data,
            required={"position", "ccd"},
            allowed={"position", "ccd"},
            location=location,
        )
        position = data["position"]
        if type(position) is not int or position < 0:
            raise PreparedInputError(
                f"{location}.position must be a non-negative integer"
            )
        return cls(
            position=position,
            ccd=_expect_nonempty_string(data["ccd"], f"{location}.ccd"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"position": self.position, "ccd": self.ccd}


@dataclass(frozen=True)
class PreparedEntity:
    """One chemical entity, optionally instantiated under several chain IDs."""

    kind: str
    ids: tuple[str, ...]
    sequence: str | None = None
    modifications: tuple[PreparedModification, ...] = ()
    smiles: str | None = None
    ccd: tuple[str, ...] | None = None
    msa: str | None = None
    msa_path: Path | None = None
    paired_msa: str | None = None
    paired_msa_path: Path | None = None
    unpaired_msa: str | None = None
    unpaired_msa_path: Path | None = None

    @property
    def msa_mode(self) -> MSAMode:
        if self.msa is not None or self.msa_path is not None:
            return "native"
        if any(
            value is not None
            for value in (
                self.paired_msa,
                self.paired_msa_path,
                self.unpaired_msa,
                self.unpaired_msa_path,
            )
        ):
            return "split"
        return "none"

    @classmethod
    def from_dict(cls, value: Any, index: int) -> PreparedEntity:
        location = f"sequences[{index}]"
        data = _expect_mapping(value, location)
        kind = data.get("type")
        if not isinstance(kind, str) or kind not in _ENTITY_TYPES:
            raise PreparedInputError(
                f"{location}.type must be one of: {', '.join(sorted(_ENTITY_TYPES))}"
            )
        required = {"type", "id"} | ({"smiles"} if kind == "ligand" else {"sequence"})
        if kind == "ligand":
            # A CCD-only ligand intentionally has no smiles field.
            required = {"type", "id"}
        _expect_fields(
            data,
            required=required,
            allowed=_ENTITY_FIELDS[kind],
            location=location,
        )
        ids = _parse_ids(data["id"], f"{location}.id")

        if kind == "ligand":
            smiles = _optional_string(data.get("smiles"), f"{location}.smiles")
            raw_ccd = data.get("ccd")
            ccd = None
            if raw_ccd is not None:
                if not isinstance(raw_ccd, list) or not raw_ccd:
                    raise PreparedInputError(f"{location}.ccd must be a non-empty list")
                ccd = tuple(
                    _expect_nonempty_string(item, f"{location}.ccd[{ccd_index}]")
                    for ccd_index, item in enumerate(raw_ccd)
                )
            if (smiles is None) == (ccd is None):
                raise PreparedInputError(
                    f"{location} must set exactly one of smiles/ccd"
                )
            return cls(kind=kind, ids=ids, smiles=smiles, ccd=ccd)

        sequence = _expect_nonempty_string(data["sequence"], f"{location}.sequence")
        if any(character.isspace() for character in sequence):
            raise PreparedInputError(f"{location}.sequence must not contain whitespace")
        if ":" in sequence or "|" in sequence:
            raise PreparedInputError(
                f"{location}.sequence must describe one entity; use separate entries "
                "and chain IDs instead of ':' or '|' chain breaks"
            )
        if not sequence.isascii() or not sequence.isalpha() or sequence != sequence.upper():
            raise PreparedInputError(
                f"{location}.sequence must contain uppercase ASCII residue letters"
            )

        raw_modifications = data.get("modifications", [])
        if not isinstance(raw_modifications, list):
            raise PreparedInputError(f"{location}.modifications must be a list")
        modifications = tuple(
            PreparedModification.from_dict(
                item, f"{location}.modifications[{mod_index}]"
            )
            for mod_index, item in enumerate(raw_modifications)
        )
        positions = [modification.position for modification in modifications]
        if len(positions) != len(set(positions)):
            raise PreparedInputError(
                f"{location}.modifications contains duplicate positions"
            )
        if any(position >= len(sequence) for position in positions):
            raise PreparedInputError(
                f"{location}.modifications contains a position outside the sequence"
            )

        entity = cls(
            kind=kind,
            ids=ids,
            sequence=sequence,
            modifications=modifications,
        )
        if kind != "protein":
            return entity

        entity = replace(
            entity,
            msa=_optional_string(data.get("msa"), f"{location}.msa"),
            msa_path=_optional_path(data.get("msaPath"), f"{location}.msaPath"),
            paired_msa=_optional_string(
                data.get("pairedMsa"), f"{location}.pairedMsa", allow_empty=True
            ),
            paired_msa_path=_optional_path(
                data.get("pairedMsaPath"), f"{location}.pairedMsaPath"
            ),
            unpaired_msa=_optional_string(
                data.get("unpairedMsa"), f"{location}.unpairedMsa"
            ),
            unpaired_msa_path=_optional_path(
                data.get("unpairedMsaPath"), f"{location}.unpairedMsaPath"
            ),
        )
        entity._validate_msa_sources(location)
        return entity

    def _validate_msa_sources(self, location: str) -> None:
        native_count = sum(value is not None for value in (self.msa, self.msa_path))
        paired_count = sum(
            value is not None for value in (self.paired_msa, self.paired_msa_path)
        )
        unpaired_count = sum(
            value is not None for value in (self.unpaired_msa, self.unpaired_msa_path)
        )
        if native_count > 1:
            raise PreparedInputError(f"{location} can set only one of msa/msaPath")
        if paired_count > 1:
            raise PreparedInputError(
                f"{location} can set only one of pairedMsa/pairedMsaPath"
            )
        if unpaired_count > 1:
            raise PreparedInputError(
                f"{location} can set only one of unpairedMsa/unpairedMsaPath"
            )
        if native_count and (paired_count or unpaired_count):
            raise PreparedInputError(
                f"{location} cannot mix msa/msaPath with paired and unpaired MSA fields"
            )
        if bool(paired_count) != bool(unpaired_count):
            raise PreparedInputError(
                f"{location} split MSA mode requires both paired and unpaired "
                "MSA sources"
            )

    def to_dict(self) -> dict[str, Any]:
        entity: dict[str, Any] = {
            "type": self.kind,
            "id": self.ids[0] if len(self.ids) == 1 else list(self.ids),
        }
        if self.kind == "ligand":
            if self.smiles is not None:
                entity["smiles"] = self.smiles
            else:
                entity["ccd"] = list(self.ccd or ())
            return entity

        entity["sequence"] = self.sequence
        if self.modifications:
            entity["modifications"] = [item.to_dict() for item in self.modifications]
        resource_fields = (
            ("msa", self.msa),
            ("msaPath", self.msa_path),
            ("pairedMsa", self.paired_msa),
            ("pairedMsaPath", self.paired_msa_path),
            ("unpairedMsa", self.unpaired_msa),
            ("unpairedMsaPath", self.unpaired_msa_path),
        )
        for key, value in resource_fields:
            if value is not None:
                entity[key] = os.fspath(value) if isinstance(value, Path) else value
        return entity

    def resolved(self, manifest_path: Path) -> PreparedEntity:
        return replace(
            self,
            msa_path=(
                None
                if self.msa_path is None
                else _resolve_path(self.msa_path, manifest_path)
            ),
            paired_msa_path=(
                None
                if self.paired_msa_path is None
                else _resolve_path(self.paired_msa_path, manifest_path)
            ),
            unpaired_msa_path=(
                None
                if self.unpaired_msa_path is None
                else _resolve_path(self.unpaired_msa_path, manifest_path)
            ),
        )


@dataclass(frozen=True)
class PreparedInput:
    """One validated ESMFold2 prediction target."""

    version: int
    name: str
    sequences: tuple[PreparedEntity, ...]

    @classmethod
    def from_dict(cls, value: Any) -> PreparedInput:
        data = _expect_mapping(value, "prepared input")
        _expect_fields(
            data,
            required=_TOP_LEVEL_REQUIRED_FIELDS,
            allowed=_TOP_LEVEL_REQUIRED_FIELDS | _TOP_LEVEL_OPTIONAL_FIELDS,
            location="prepared input",
        )
        version = data["version"]
        if type(version) is not int or version != PREPARED_INPUT_VERSION:
            raise PreparedInputError(
                f"version must be the integer {PREPARED_INPUT_VERSION}"
            )
        raw_sequences = data["sequences"]
        if not isinstance(raw_sequences, list) or not raw_sequences:
            raise PreparedInputError("sequences must be a non-empty list")
        sequences = tuple(
            PreparedEntity.from_dict(item, index)
            for index, item in enumerate(raw_sequences)
        )
        seen_ids: set[str] = set()
        for index, entity in enumerate(sequences):
            duplicates = seen_ids.intersection(entity.ids)
            if duplicates:
                raise PreparedInputError(
                    f"sequences[{index}] reuses chain IDs: "
                    f"{', '.join(sorted(duplicates))}"
                )
            seen_ids.update(entity.ids)
        return cls(
            version=version,
            name=_validate_safe_name(data["name"], "name"),
            sequences=sequences,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "sequences": [entity.to_dict() for entity in self.sequences],
        }

    def resolved(self, manifest_path: str | Path) -> PreparedInput:
        path = Path(manifest_path)
        return replace(
            self,
            sequences=tuple(entity.resolved(path) for entity in self.sequences),
        )

    def validate_resources(self, manifest_path: str | Path) -> PreparedInput:
        resolved = self.resolved(manifest_path)
        for index, entity in enumerate(resolved.sequences):
            for field, path in (
                ("msaPath", entity.msa_path),
                ("pairedMsaPath", entity.paired_msa_path),
                ("unpairedMsaPath", entity.unpaired_msa_path),
            ):
                if path is not None and not path.is_file():
                    raise PreparedInputError(
                        f"sequences[{index}].{field} does not exist or is not a "
                        f"file: {path}"
                    )
        return resolved


def load_prepared_input(path: str | Path) -> PreparedInput:
    """Read and validate one wrapper manifest without loading model resources."""
    source = Path(path).expanduser()
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as error:
        raise PreparedInputError(f"Invalid JSON in {source}: {error}") from error
    except OSError as error:
        raise PreparedInputError(
            f"Cannot read prepared input {source}: {error}"
        ) from error
    return PreparedInput.from_dict(value)


def prepared_input_path(output_dir: str | Path, name: str) -> Path:
    """Return the canonical ``<output>/<name>/<name>_data.json`` location."""
    safe_name = _validate_safe_name(name, "name")
    output_root = Path(output_dir).expanduser().resolve()
    return output_root / safe_name / f"{safe_name}_data.json"


def write_prepared_input(prepared: PreparedInput, path: str | Path) -> Path:
    """Atomically write a canonical, revalidated manifest."""
    validated = PreparedInput.from_dict(prepared.to_dict())
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(validated.to_dict(), handle, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
