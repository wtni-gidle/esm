"""Unit tests for the ESMFold2 first-layer wrapper manifest."""

import json

import pytest

from esm.esmfold2_wrapper.input import (
    PreparedInput,
    PreparedInputError,
    load_prepared_input,
    prepared_input_path,
    write_prepared_input,
)


def minimal_input(**protein_fields):
    protein = {"type": "protein", "id": "A", "sequence": "ACDE"}
    protein.update(protein_fields)
    return {"version": 1, "name": "target-1", "sequences": [protein]}


def test_query_only_protein_round_trips():
    prepared = PreparedInput.from_dict(minimal_input())

    assert prepared.sequences[0].ids == ("A",)
    assert prepared.sequences[0].msa_mode == "none"
    assert prepared.to_dict() == minimal_input()


def test_multicopy_and_supported_entity_types_round_trip():
    value = {
        "version": 1,
        "name": "complex",
        "sequences": [
            {"type": "protein", "id": ["A", "B"], "sequence": "ACDE"},
            {
                "type": "rna",
                "id": "R",
                "sequence": "ACGU",
                "modifications": [{"position": 1, "ccd": "5MC"}],
            },
            {"type": "dna", "id": "D", "sequence": "ACGT"},
            {"type": "ligand", "id": "L", "ccd": ["ATP"]},
            {"type": "ligand", "id": "M", "smiles": "[Mg+2]"},
        ],
    }

    assert PreparedInput.from_dict(value).to_dict() == value


@pytest.mark.parametrize(
    "fields,mode",
    [
        ({"msa": ">q\nACDE\n"}, "native"),
        ({"msaPath": "query.a3m.zst"}, "native"),
        (
            {
                "pairedMsa": "",
                "unpairedMsa": ">q\nACDE\n",
            },
            "split",
        ),
        (
            {
                "pairedMsaPath": "paired.a3m.zst",
                "unpairedMsaPath": "unpaired.a3m.zst",
            },
            "split",
        ),
    ],
)
def test_msa_modes(fields, mode):
    entity = PreparedInput.from_dict(minimal_input(**fields)).sequences[0]
    assert entity.msa_mode == mode


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"msa": ">q\nACDE\n", "msaPath": "x.a3m"}, "only one of msa/msaPath"),
        (
            {"pairedMsa": "", "unpairedMsa": ">q\nACDE\n", "msaPath": "x.a3m"},
            "cannot mix",
        ),
        ({"pairedMsa": ""}, "requires both paired and unpaired"),
        ({"unpairedMsa": ">q\nACDE\n"}, "requires both paired and unpaired"),
    ],
)
def test_rejects_ambiguous_msa_sources(fields, message):
    with pytest.raises(PreparedInputError, match=message):
        PreparedInput.from_dict(minimal_input(**fields))


def test_rejects_duplicate_chain_ids_across_entities():
    value = minimal_input()
    value["sequences"].append({"type": "dna", "id": "A", "sequence": "ACGT"})
    with pytest.raises(PreparedInputError, match="reuses chain IDs"):
        PreparedInput.from_dict(value)


def test_rejects_unknown_fields_and_invalid_names():
    with pytest.raises(PreparedInputError, match="unknown fields"):
        PreparedInput.from_dict(minimal_input(templates=[]))
    with pytest.raises(PreparedInputError, match="ASCII letters"):
        PreparedInput.from_dict({**minimal_input(), "name": "../escape"})


@pytest.mark.parametrize("invalid_type", [[], {}])
def test_rejects_non_string_entity_type(invalid_type):
    value = minimal_input()
    value["sequences"][0]["type"] = invalid_type
    with pytest.raises(PreparedInputError, match="type must be one of"):
        PreparedInput.from_dict(value)


def test_rejects_lowercase_sequences_and_unsupported_modification_smiles():
    with pytest.raises(PreparedInputError, match="uppercase ASCII"):
        PreparedInput.from_dict({
            **minimal_input(),
            "sequences": [{"type": "protein", "id": "A", "sequence": "AcDE"}],
        })
    with pytest.raises(PreparedInputError, match="unknown fields: smiles"):
        PreparedInput.from_dict(minimal_input(modifications=[{
            "position": 1,
            "ccd": "MSE",
            "smiles": "unused",
        }]))


@pytest.mark.parametrize("sequence", ["AC:DE", "AC|DE"])
def test_rejects_chain_break_shortcuts(sequence):
    with pytest.raises(PreparedInputError, match="separate entries"):
        PreparedInput.from_dict(
            {
                **minimal_input(),
                "sequences": [
                    {"type": "protein", "id": "A", "sequence": sequence}
                ],
            }
        )


def test_paths_resolve_relative_to_manifest_and_are_checked(tmp_path):
    msa = tmp_path / "msas" / "query.a3m"
    msa.parent.mkdir()
    msa.write_text(">q\nACDE\n")
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps(minimal_input(msaPath="msas/query.a3m")))

    prepared = load_prepared_input(manifest).validate_resources(manifest)
    assert prepared.sequences[0].msa_path == msa.resolve()

    msa.unlink()
    with pytest.raises(PreparedInputError, match="does not exist"):
        load_prepared_input(manifest).validate_resources(manifest)


def test_atomic_writer_uses_canonical_job_path(tmp_path):
    prepared = PreparedInput.from_dict(minimal_input())
    path = prepared_input_path(tmp_path, prepared.name)

    assert write_prepared_input(prepared, path) == path
    assert json.loads(path.read_text()) == minimal_input()
    assert not list(path.parent.glob(".*.tmp"))
