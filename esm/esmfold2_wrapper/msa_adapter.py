"""Adapt native or Boltz-style A3M text to ESMFold2's native :class:`MSA`."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from io import StringIO
from typing import TYPE_CHECKING

from esm.utils.parsing import FastaEntry

if TYPE_CHECKING:
    from esm.utils.msa import MSA

_A3M_CHARACTERS = frozenset(string.ascii_letters + "-.")
_ESMFOLD_KEY = re.compile(r"key=(-?\d+)")


class PreparedMSAError(ValueError):
    """Raised when an MSA resource cannot represent the declared protein."""


def validate_split_paired_depths(
    components: Sequence[tuple[str, str]],
) -> None:
    """Require Boltz component files to share their original positional depth."""
    if len(components) < 2:
        return
    depths = [
        (entity_id, len(parse_a3m_records(text, f"{entity_id}.pairedMsa")))
        for entity_id, text in components
    ]
    expected = depths[0][1]
    if any(depth != expected for _, depth in depths[1:]):
        detail = ", ".join(f"{entity_id}={depth}" for entity_id, depth in depths)
        raise PreparedMSAError(
            "Boltz-style paired MSA files must have the same original row count "
            f"across split-mode protein entities; found {detail}"
        )


def parse_a3m_records(text: str, location: str) -> tuple[FastaEntry, ...]:
    """Parse an A3M while preserving headers, insertions and original row order."""
    if not isinstance(text, str) or "\x00" in text:
        raise PreparedMSAError(f"{location} must be UTF-8 A3M text")
    if not text.strip():
        return ()

    records: list[FastaEntry] = []
    header: str | None = None
    sequence_lines: list[str] = []

    def finish_record() -> None:
        if header is None:
            return
        sequence = "".join(sequence_lines)
        if not sequence:
            raise PreparedMSAError(
                f"{location} record {len(records)} has an empty sequence"
            )
        invalid = sorted(set(sequence) - _A3M_CHARACTERS)
        if invalid:
            raise PreparedMSAError(
                f"{location} record {len(records)} contains invalid A3M "
                f"characters: {''.join(invalid)!r}"
            )
        records.append(FastaEntry(header, sequence))

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            finish_record()
            header = line[1:].strip()
            if not header:
                raise PreparedMSAError(
                    f"{location} line {line_number} has an empty FASTA header"
                )
            sequence_lines = []
            continue
        if header is None:
            raise PreparedMSAError(
                f"{location} line {line_number} contains sequence before a header"
            )
        if any(character.isspace() for character in line):
            raise PreparedMSAError(
                f"{location} line {line_number} contains whitespace inside a sequence"
            )
        sequence_lines.append(line)

    finish_record()
    if not records:
        raise PreparedMSAError(f"{location} contains no A3M records")
    return tuple(records)


def _aligned_sequence(sequence: str) -> str:
    """Remove lowercase/dot insertions but retain aligned residues and gaps."""
    return "".join(
        character
        for character in sequence
        if not character.islower() and character != "."
    )


def _validate_records(
    records: tuple[FastaEntry, ...], query_sequence: str, location: str
) -> None:
    aligned_query = _aligned_sequence(records[0].sequence).replace("-", "")
    if aligned_query.upper() != query_sequence.upper():
        raise PreparedMSAError(
            f"{location} query does not match the JSON protein sequence: "
            f"expected {query_sequence!r}, found {aligned_query!r}"
        )
    expected_width = len(query_sequence)
    for index, record in enumerate(records):
        width = len(_aligned_sequence(record.sequence))
        if width != expected_width:
            raise PreparedMSAError(
                f"{location} record {index} has aligned width {width}; "
                f"expected {expected_width}"
            )


def _is_all_gap(sequence: str) -> bool:
    """Match Boltz's padding-row rule exactly, before insertion processing."""
    return bool(sequence) and set(sequence) == {"-"}


def _validate_native_keys(records: tuple[FastaEntry, ...]) -> None:
    for index, record in enumerate(records):
        matches = _ESMFOLD_KEY.findall(record.header)
        if len(matches) > 1:
            raise PreparedMSAError(
                f"msa record {index} contains more than one ESMFold2 key"
            )
        if matches and int(matches[0]) < -1:
            raise PreparedMSAError(
                f"msa record {index} has invalid key={matches[0]}; "
                "keys must be -1 or non-negative"
            )


def _records_to_a3m(records: list[FastaEntry]) -> str:
    return "\n".join(
        line
        for record in records
        for line in (f">{record.header}", record.sequence)
    ) + "\n"


def split_a3m_to_keyed_text(
    *, paired_a3m: str, unpaired_a3m: str, query_sequence: str
) -> str:
    """Merge Boltz component A3Ms into ESMFold2's ``key=N`` A3M convention.

    Paired keys are the zero-based row positions in the original paired file.
    Padding rows are removed without renumbering, keeping keys aligned across
    chains generated by the same Boltz pairing search.
    """
    paired = parse_a3m_records(paired_a3m, "pairedMsa")
    unpaired = parse_a3m_records(unpaired_a3m, "unpairedMsa")
    if not unpaired:
        raise PreparedMSAError("unpairedMsa must contain at least the query record")
    if paired:
        _validate_records(paired, query_sequence, "pairedMsa")
    _validate_records(unpaired, query_sequence, "unpairedMsa")

    merged: list[FastaEntry] = []
    for row_index, record in enumerate(paired):
        if _is_all_gap(record.sequence):
            continue
        merged.append(
            FastaEntry(
                f"paired_row_{row_index} key={row_index}",
                record.sequence,
            )
        )

    unpaired_start = 1 if paired else 0
    for row_index, record in enumerate(unpaired[unpaired_start:], start=unpaired_start):
        merged.append(
            FastaEntry(
                f"unpaired_row_{row_index} key=-1",
                record.sequence,
            )
        )

    if not merged:
        # The validated unpaired query guarantees this is defensive only.
        raise PreparedMSAError("paired/unpaired merge produced no MSA records")
    return _records_to_a3m(merged)


def split_a3m_to_esmfold2_msa(
    *, paired_a3m: str, unpaired_a3m: str, query_sequence: str
) -> MSA:
    """Return an insertion-stripped native MSA with deletion features retained."""
    keyed = split_a3m_to_keyed_text(
        paired_a3m=paired_a3m,
        unpaired_a3m=unpaired_a3m,
        query_sequence=query_sequence,
    )
    # Import lazily and only after pure validation: esm.utils.msa currently
    # reaches Torch via esm.utils.misc. Data-only never takes this path.
    from esm.utils.msa import MSA

    return MSA.from_a3m(StringIO(keyed), remove_insertions=True)


def validate_native_a3m(*, a3m: str, query_sequence: str) -> None:
    """Validate the query and match-column width of a native keyed/standard A3M."""
    records = parse_a3m_records(a3m, "msa")
    if not records:
        raise PreparedMSAError("msa must contain at least the query record")
    _validate_records(records, query_sequence, "msa")
    _validate_native_keys(records)


def native_a3m_to_esmfold2_msa(*, a3m: str, query_sequence: str) -> MSA:
    """Validate a native keyed/standard A3M and retain its deletion features."""
    from esm.utils.msa import MSA

    records = parse_a3m_records(a3m, "msa")
    validate_native_a3m(a3m=a3m, query_sequence=query_sequence)
    return MSA.from_a3m(StringIO(_records_to_a3m(list(records))), remove_insertions=True)
