"""Magic-byte text compression helpers for portable prepared resources."""

from __future__ import annotations

import gzip
import lzma
import os
import uuid
from pathlib import Path
from typing import cast

import zstandard as zstd

_GZIP_MAGIC = b"\x1f\x8b"
_XZ_MAGIC = b"\xfd7zXZ\x00"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def read_text_auto(path: str | Path) -> str:
    """Read UTF-8 plain, gzip, xz or zstd text based on content, not suffix."""
    source = Path(path)
    try:
        with source.open("rb") as raw_file:
            header = raw_file.read(6)
            raw_file.seek(0)
            if header.startswith(_GZIP_MAGIC):
                with gzip.open(raw_file, "rt", encoding="utf-8") as text_file:
                    return text_file.read()
            if header.startswith(_XZ_MAGIC):
                with lzma.open(raw_file, "rt", encoding="utf-8") as text_file:
                    return cast(str, text_file.read())
            if header.startswith(_ZSTD_MAGIC):
                with zstd.open(raw_file, "rt", encoding="utf-8") as text_file:
                    return cast(str, text_file.read())
            return raw_file.read().decode("utf-8")
    except ImportError:
        raise
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise ValueError(
            f"Cannot read UTF-8 text resource {source}: {error}"
        ) from error


def write_zstd_text(path: str | Path, text: str) -> Path:
    """Atomically write UTF-8 text as a zstd frame."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(zstd.ZstdCompressor().compress(text.encode("utf-8")))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
