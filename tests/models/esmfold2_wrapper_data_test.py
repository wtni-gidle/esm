"""Tests for portable ESMFold2 prepared MSA bundles."""

import gzip
import json
import lzma

import pytest
import zstandard as zstd

from esm.esmfold2_wrapper.compression import read_text_auto, write_zstd_text
from esm.esmfold2_wrapper.data import prepare_data_bundle
from esm.esmfold2_wrapper.input import load_prepared_input


def write_manifest(path, sequences, name="target"):
    path.write_text(json.dumps({"version": 1, "name": name, "sequences": sequences}))


def test_inline_split_msa_is_kept_as_two_portable_authoritative_resources(tmp_path):
    source = tmp_path / "input.json"
    paired = ">query\nACDE\n>paired\nAC-E\n"
    unpaired = ">query\nACDE\n>single\nA-DE\n"
    write_manifest(
        source,
        [
            {
                "type": "protein",
                "id": ["A", "B"],
                "sequence": "ACDE",
                "pairedMsa": paired,
                "unpairedMsa": unpaired,
            }
        ],
    )
    output = tmp_path / "out" / "target" / "target_data.json"

    prepare_data_bundle(source, output, compress_fold_input=True)

    stored = json.loads(output.read_text())
    protein = stored["sequences"][0]
    assert protein["pairedMsaPath"] == "msas/target__A_pairedmsa.a3m.zst"
    assert protein["unpairedMsaPath"] == "msas/target__A_unpairedmsa.a3m.zst"
    assert "pairedMsa" not in protein and "unpairedMsa" not in protein
    assert read_text_auto(output.parent / protein["pairedMsaPath"]) == paired
    assert read_text_auto(output.parent / protein["unpairedMsaPath"]) == unpaired
    assert not list(output.parent.glob("**/*_msa.a3m.zst"))


def test_empty_paired_is_materialized_instead_of_becoming_missing(tmp_path):
    source = tmp_path / "input.json"
    write_manifest(
        source,
        [
            {
                "type": "protein",
                "id": "A",
                "sequence": "ACDE",
                "pairedMsa": "",
                "unpairedMsa": ">query\nACDE\n",
            }
        ],
    )
    output = tmp_path / "out/target/target_data.json"

    prepare_data_bundle(source, output, compress_fold_input=True)

    prepared = load_prepared_input(output).validate_resources(output)
    protein = prepared.sequences[0]
    assert protein.msa_mode == "split"
    assert read_text_auto(protein.paired_msa_path) == ""


def test_gzip_source_is_detected_by_magic_and_rewritten_as_zstd(tmp_path):
    source_msa = tmp_path / "misleading.a3m"
    source_msa.write_bytes(gzip.compress(b">query\nACDE\n"))
    source = tmp_path / "input.json"
    write_manifest(
        source,
        [
            {
                "type": "protein",
                "id": "A",
                "sequence": "ACDE",
                "msaPath": source_msa.name,
            }
        ],
    )
    output = tmp_path / "out/target/target_data.json"

    prepare_data_bundle(source, output, compress_fold_input=True)

    protein = load_prepared_input(output).validate_resources(output).sequences[0]
    assert protein.msa_path.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
    assert read_text_auto(protein.msa_path) == ">query\nACDE\n"


def test_compression_reader_uses_magic_instead_of_filename_suffix(tmp_path):
    expected = ">query\nACDE\n"
    plain_with_zstd_suffix = tmp_path / "plain.a3m.zst"
    gzip_with_plain_suffix = tmp_path / "gzip.a3m"
    xz_with_plain_suffix = tmp_path / "xz.a3m"
    zstd_with_plain_suffix = tmp_path / "zstd.a3m"
    plain_with_zstd_suffix.write_text(expected)
    gzip_with_plain_suffix.write_bytes(gzip.compress(expected.encode()))
    xz_with_plain_suffix.write_bytes(lzma.compress(expected.encode()))
    write_zstd_text(zstd_with_plain_suffix, expected)

    for path in (
        plain_with_zstd_suffix,
        gzip_with_plain_suffix,
        xz_with_plain_suffix,
        zstd_with_plain_suffix,
    ):
        assert read_text_auto(path) == expected


def test_zstd_reader_streams_unknown_content_size_past_128_kib(tmp_path):
    expected = "x" * 131_073
    source = tmp_path / "unknown-size.a3m.zst"
    source.write_bytes(
        zstd.ZstdCompressor(write_content_size=False).compress(expected.encode("utf-8"))
    )

    assert read_text_auto(source) == expected


def test_corrupt_zstd_error_names_the_resource(tmp_path):
    source = tmp_path / "corrupt.a3m.zst"
    source.write_bytes(b"\x28\xb5\x2f\xfdcorrupt")

    with pytest.raises(
        ValueError, match=r"Cannot read UTF-8 text resource .*corrupt\.a3m\.zst:"
    ):
        read_text_auto(source)


def test_bundle_remains_valid_after_moving_the_job_directory(tmp_path):
    source = tmp_path / "input.json"
    write_manifest(
        source,
        [{"type": "protein", "id": "A", "sequence": "ACDE", "msa": ">query\nACDE\n"}],
    )
    original = tmp_path / "out/target/target_data.json"
    prepare_data_bundle(source, original, compress_fold_input=True)

    moved_job = tmp_path / "moved"
    original.parent.rename(moved_job)
    moved_manifest = moved_job / "target_data.json"

    prepared = load_prepared_input(moved_manifest).validate_resources(moved_manifest)
    assert read_text_auto(prepared.sequences[0].msa_path) == ">query\nACDE\n"


def test_all_inputs_are_validated_before_bundle_is_written(tmp_path):
    source = tmp_path / "input.json"
    write_manifest(
        source,
        [
            {"type": "protein", "id": "A", "sequence": "ACDE", "msa": ">q\nACDE\n"},
            {"type": "protein", "id": "B", "sequence": "FGHI", "msa": ">q\nBAD\n"},
        ],
    )
    output = tmp_path / "out/target/target_data.json"

    with pytest.raises(ValueError, match="does not match"):
        prepare_data_bundle(source, output, compress_fold_input=True)

    assert not output.parent.exists()


def test_split_entities_require_matching_original_paired_depths(tmp_path):
    source = tmp_path / "input.json"
    write_manifest(
        source,
        [
            {
                "type": "protein",
                "id": "A",
                "sequence": "ACDE",
                "pairedMsa": ">q\nACDE\n>p\nAC-E\n",
                "unpairedMsa": ">q\nACDE\n",
            },
            {
                "type": "protein",
                "id": "B",
                "sequence": "FGHI",
                "pairedMsa": ">q\nFGHI\n",
                "unpairedMsa": ">q\nFGHI\n",
            },
        ],
    )

    with pytest.raises(ValueError, match="same original row count"):
        prepare_data_bundle(source, tmp_path / "out/target/target_data.json", compress_fold_input=True)


def test_case_insensitive_resource_name_collisions_are_rejected(tmp_path):
    source = tmp_path / "input.json"
    write_manifest(
        source,
        [
            {"type": "protein", "id": "A", "sequence": "ACDE", "msa": ">q\nACDE\n"},
            {"type": "protein", "id": "a", "sequence": "ACDE", "msa": ">q\nACDE\n"},
        ],
    )

    with pytest.raises(ValueError, match="case-insensitive"):
        prepare_data_bundle(source, tmp_path / "out/target/target_data.json", compress_fold_input=True)


def test_existing_msas_symlink_cannot_escape_the_job_directory(tmp_path):
    source = tmp_path / "input.json"
    write_manifest(
        source,
        [{"type": "protein", "id": "A", "sequence": "ACDE", "msa": ">q\nACDE\n"}],
    )
    job = tmp_path / "out/target"
    job.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (job / "msas").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        prepare_data_bundle(source, job / "target_data.json", compress_fold_input=True)
    assert not list(outside.iterdir())
