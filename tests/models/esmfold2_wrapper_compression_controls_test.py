"""Input and detailed-confidence compression are independent publication options."""
import json
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from esm.esmfold2_wrapper import cli, inference
from esm.esmfold2_wrapper.compression import read_text_auto
from esm.esmfold2_wrapper.data import prepare_data_bundle
from esm.esmfold2_wrapper.input import load_prepared_input
from esm.esmfold2_wrapper.outputs import publish_inference_results, seed_outputs_complete


def _source(tmp_path, mode):
    path = tmp_path / "input.json"
    msa = {"msa": ">query\nA\n>hit\nW\n"} if mode == "native" else {
        "pairedMsa": "", "unpairedMsa": ">query\nA\n>hit\nW\n"}
    path.write_text(json.dumps({"version": 1, "name": "job", "sequences": [
        {"type": "protein", "id": "A", "sequence": "A", **msa}]}))
    return path


@pytest.mark.parametrize("mode", ["native", "split"])
@pytest.mark.parametrize("compressed", [None, False, True])
def test_prepared_compressed_to_selected_format_roundtrip(tmp_path, mode, compressed):
    source = _source(tmp_path, mode)
    old = tmp_path / "old/job_data.json"
    prepare_data_bundle(source, old, compress_fold_input=True)
    target = tmp_path / "new/job_data.json"
    options = {} if compressed is None else {"compress_fold_input": compressed}
    prepare_data_bundle(old, target, **options)
    stored = json.loads(target.read_text())["sequences"][0]
    field = "msaPath" if mode == "native" else "unpairedMsaPath"
    stem = "msa" if mode == "native" else "unpairedmsa"
    assert stored[field] == f"msas/job__A_{stem}.a3m" + (".zst" if compressed else "")
    assert read_text_auto(target.parent / stored[field]) == ">query\nA\n>hit\nW\n"
    restored = load_prepared_input(target).validate_resources(target).sequences[0]
    assert restored.msa_mode == mode


def _result():
    return SimpleNamespace(
        complex=SimpleNamespace(to_mmcif=lambda: "data_sample\n#\n", plddt=np.array([0.5]), metadata=SimpleNamespace(chain_lookup={0: "A"})),
        plddt=np.array([0.5], dtype=np.float32), pae=np.array([[0.25]], dtype=np.float32),
        pde=np.array([[0.75]], dtype=np.float32), ptm=0.5, iptm=0.25,
        pair_chains_iptm=np.array([[0.25]]), output_embedding_pair_pooled=np.array([[1.0]]),
        output_embedding_sequence=None, residue_index=np.array([0]), entity_id=np.array([0]))


@pytest.mark.parametrize("compressed", [None, False, True])
def test_full_confidence_shapes_switch_cleanup_and_resume(tmp_path, compressed):
    for selection in (not bool(compressed), compressed):
        options = {} if selection is None else {"compress_full_confidence": selection}
        paths = publish_inference_results([_result()], predictions_dir=tmp_path, seed=7, include_embeddings=True, **options)[0]
        for kind, path, expected in (
            ("plddt", paths.plddt_path, {"plddt": [0.5], "structure_token_plddt": [0.5]}),
            ("pae", paths.pae_path, {"pae": [[0.25]]}),
            ("pde", paths.pde_path, {"pde": [[0.75]]}),
        ):
            assert path.name == f"{kind}_seed-7_sample-0." + ("npz" if selection else "json")
            assert not path.with_suffix(".json" if selection else ".npz").exists()
            if selection:
                with ZipFile(path) as archive:
                    assert all(item.compress_type == ZIP_DEFLATED for item in archive.infolist())
                with np.load(path) as archive:
                    actual = dict(archive)
            else:
                actual = json.loads(path.read_text())
            assert set(actual) == set(expected)
            for key, value in expected.items():
                np.testing.assert_array_equal(actual[key], value)
        assert (tmp_path / "embeddings/seed-7_embeddings.npz").is_file()
        assert seed_outputs_complete(tmp_path, seed=7, sample_count=1, include_embeddings=True, **options)
    paths.pae_path.write_text("")
    assert not seed_outputs_complete(tmp_path, seed=7, sample_count=1, **options)


@pytest.mark.parametrize("write", [None, False, True])
def test_cli_inference_skip_still_refreshes_default_plain_input(tmp_path, monkeypatch, write):
    source = _source(tmp_path, "split")
    output = tmp_path / "output"
    for rel in ("models/seed-7_sample-0_model.cif", "summary_confidences/seed-7_sample-0_summary_confidences.json", "full_data/plddt_seed-7_sample-0.json", "full_data/pae_seed-7_sample-0.json", "full_data/pde_seed-7_sample-0.json"):
        path = output / "job" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("complete without parsing")
    def forbidden(*args, **kwargs):
        pytest.fail("completed seed must skip model loading")
    monkeypatch.setattr(inference, "load_esmfold2_model", forbidden)
    args = ["predict", "-i", str(source), "-o", str(output), "-D", "false", "-S", "true", "-r", "7", "-n", "1"]
    if write is not None:
        args += ["-J", str(write).lower()]
    assert cli.main(args) == 0
    published = output / "job/job_data.json"
    assert published.exists() is (write is not False)
    if published.exists():
        resource = output / "job/msas/job__A_unpairedmsa.a3m"
        assert resource.read_text().endswith(">hit\nW\n")
        source.write_text(source.read_text().replace("W\\n", "G\\n"))
        assert cli.main(args) == 0
        assert resource.read_text().endswith(">hit\nG\n")
