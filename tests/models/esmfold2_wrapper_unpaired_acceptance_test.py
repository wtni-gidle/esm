"""Replacement contract through the real two-chain MSA feature consumer.

Breaks caught: stale resource reuse, renumbering paired rows after removing a gap,
lost A3M insertions, changed paired input, or public writes with J=false.
Only model loading/folding is substituted; token metadata is a tiny hand-built
standard-protein fixture so CCD/conformer preparation is outside this test.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from esm.esmfold2_wrapper import inference
from esm.esmfold2_wrapper.compression import read_text_auto, write_zstd_text
from esm.esmfold2_wrapper.workflow import run_prepared_workflow
from esm.models.esmfold2.paired_msa import protein_letter_to_res_type
from esm.models.esmfold2.prepare_input import ChainInfo, TokenInfo, compute_msa_features
from tests.models.esmfold2_wrapper_workflow_test import fake_result


def _native_features(structure_input):
    alphabet = protein_letter_to_res_type()
    chains = [ChainInfo("A", 0, 0, 0, 0), ChainInfo("B", 1, 1, 0, 0)]
    tokens = [
        TokenInfo(
            token_index=4 * asym + pos, residue_index=pos,
            residue_name=residue, mol_type=0, res_type=alphabet[residue],
            input_id=0, asym_id=asym, sym_id=0, entity_id=asym,
            atom_start=0, atom_count=0,
        )
        for asym, sequence in enumerate(("ACDE", "FGHI"))
        for pos, residue in enumerate(sequence)
    ]
    features = compute_msa_features(structure_input, chains, tokens)
    decode = {value: key for key, value in alphabet.items()}
    rows = ["".join(decode[int(value)] for value in row)
            for row in features["msa"]]
    return rows, features


@pytest.mark.parametrize("data", [False, True])
@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("new_path", [False, True])
def test_prepared_unpaired_replacement_reaches_two_chain_features(
    tmp_path, monkeypatch, data, write, new_path
):
    source = tmp_path / "request.json"
    source.write_text(json.dumps({
        "version": 1, "name": "job", "sequences": [
            {"type": "protein", "id": "A", "sequence": "ACDE",
             "pairedMsa": ">q\nACDE\n>gap\n----\n>pair\nAC-E\n",
             "unpairedMsa": ">q\nACDE\n>old\nA-DE\n"},
            {"type": "protein", "id": "B", "sequence": "FGHI",
             "pairedMsa": ">q\nFGHI\n>only_b\nF-HI\n>pair\nFG-I\n",
             "unpairedMsa": ">q\nFGHI\n>unpaired_b\nFGH-\n"},
        ],
    }))
    output = tmp_path / "out"
    result = run_prepared_workflow(
        source, output, run_inference=False, write_input_json=True
    )
    prepared = result.prepared_path
    job = prepared.parent
    payload = json.loads(prepared.read_text())
    paired_files = [job / item["pairedMsaPath"] for item in payload["sequences"]]
    paired_bytes = [path.read_bytes() for path in paired_files]
    observed = []
    model = SimpleNamespace(
        config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=True)),
        msa_encoder=object(),
    )

    class Builder:
        def fold(self, loaded_model, structure_input, **kwargs):
            observed.append(_native_features(structure_input))
            return [fake_result(kwargs["seed"], 0)]

    monkeypatch.setattr(inference, "load_esmfold2_model", lambda *a, **k: model)
    monkeypatch.setattr(inference, "_new_input_builder", Builder)
    options = dict(seeds=7, num_diffusion_samples=1, skip=False, device="cpu")
    run_prepared_workflow(
        prepared, output, run_data_pipeline=False, write_input_json=False, **options
    )
    replacement = ">q\nACDE\n>new\nAxxCD-\n"
    changed = job / payload["sequences"][0]["unpairedMsaPath"]
    if new_path:
        changed = job / "msas/replacement.a3m.zst"
        payload["sequences"][0]["unpairedMsaPath"] = "msas/replacement.a3m.zst"
        prepared.write_text(json.dumps(payload))
    write_zstd_text(changed, replacement)
    before = {p: p.read_bytes() for p in [prepared, *job.glob("msas/*")]}
    before_paths = {path.relative_to(job) for path in job.rglob("*")}
    run_prepared_workflow(
        prepared, output, run_data_pipeline=data, write_input_json=write, **options
    )
    assert observed[0][0] == ["ACDEFGHI", "AC-EFG-I", "A-DEF-HI", "----FGH-"]
    assert observed[1][0] == ["ACDEFGHI", "AC-EFG-I", "ACD-F-HI", "----FGH-"]
    features = observed[1][1]
    assert features["has_deletion"].nonzero().tolist() == [[2, 1]]
    expected = np.zeros(8)
    expected[1] = (np.pi / 2) * np.arctan(2 / 3) / 4
    np.testing.assert_allclose(features["deletion_mean"].numpy(), expected)
    assert [path.read_bytes() for path in paired_files] == paired_bytes
    if not write:
        assert {p: p.read_bytes() for p in [prepared, *job.glob("msas/*")]} == before
        assert {path.relative_to(job) for path in job.rglob("*")} == before_paths
    else:
        current = json.loads(prepared.read_text())
        resource = current["sequences"][0]["unpairedMsaPath"]
        assert resource == "msas/job__A_unpairedmsa.a3m.zst"
        assert read_text_auto(job / resource) == replacement
        assert [item["pairedMsaPath"] for item in current["sequences"]] == [
            "msas/job__A_pairedmsa.a3m.zst", "msas/job__B_pairedmsa.a3m.zst"
        ]
