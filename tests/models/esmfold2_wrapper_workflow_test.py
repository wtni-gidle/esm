"""Tests for side-effect-free ESMFold2 wrapper planning."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import esm.esmfold2_wrapper.inference as inference_module
from esm.esmfold2_wrapper.outputs import expected_seed_samples
from esm.esmfold2_wrapper.compression import read_text_auto
from esm.esmfold2_wrapper.workflow import (
    build_workflow_plan,
    normalize_seeds,
    run_prepared_workflow,
)


def write_input(path, name="target"):
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": name,
                "sequences": [
                    {"type": "protein", "id": "A", "sequence": "ACDE"}
                ],
            }
        )
    )


def test_snapshot_targets_canonical_prepared_manifest(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)

    plan = build_workflow_plan(
        source, tmp_path / "outputs", write_input_json=True
    )

    expected = (tmp_path / "outputs/target/target_data.json").resolve()
    assert plan.prepared_path == expected
    assert plan.predictions_dir == (tmp_path / "outputs/target").resolve()
    assert not plan.job_dir.exists()


def test_outputs_use_job_root(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)

    plan = build_workflow_plan(
        source,
        tmp_path / "results",
        write_input_json=False,
    )

    job = (tmp_path / "results/target").resolve()
    assert plan.predictions_dir == job
    sample = expected_seed_samples(job, seed=7, sample_count=1)[0]
    assert sample.model_path == job / "models/seed-7_sample-0_model.cif"
    assert not job.exists()


def test_inference_only_consumes_the_given_prepared_manifest(tmp_path):
    source = tmp_path / "target_data.json"
    write_input(source)

    plan = build_workflow_plan(
        source, tmp_path / "outputs"
    )

    assert plan.prepared_path == source.resolve()


def test_rejects_no_stage_even_when_snapshot_requested(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)
    with pytest.raises(ValueError, match="At least one"):
        build_workflow_plan(
            source,
            tmp_path / "outputs",
            run_data_pipeline=False, run_inference=False, write_input_json=True,
        )


@pytest.mark.parametrize("flag", ["run_data_pipeline", "run_inference"])
def test_rejects_non_boolean_stage_flags(tmp_path, flag):
    with pytest.raises(ValueError, match="must be booleans"):
        build_workflow_plan(tmp_path / "missing.json", tmp_path / "out", **{flag: "false"})


@pytest.mark.parametrize("write_snapshot", [False, True])
def test_data_only_validates_without_loading_model(tmp_path, monkeypatch, write_snapshot):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({
        "version": 1, "name": "target", "sequences": [{
            "type": "protein", "id": "A", "sequence": "ACDE",
            "msa": ">q\nACDE\n>hit key=9\nAC-E\n",
        }],
    }))
    monkeypatch.setattr(inference_module, "load_esmfold2_model",
                        lambda *a, **k: pytest.fail("data-only must not load weights"))
    result = run_prepared_workflow(
        source, tmp_path / "out", run_data_pipeline=True, run_inference=False,
        write_input_json=write_snapshot, seeds="not an inference seed",
        num_diffusion_samples=0,
    )
    assert result.seeds == ()
    assert result.prediction_paths == ()
    if write_snapshot:
        stored = json.loads(result.prepared_path.read_text())
        assert stored["sequences"][0]["msaPath"] == "msas/target__A_msa.a3m.zst"
    else:
        assert result.prepared_path == source.resolve()
        assert not (tmp_path / "out").exists()


def test_data_only_without_writing_still_validates_msa_content(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({
        "version": 1, "name": "target", "sequences": [{
            "type": "protein", "id": "A", "sequence": "ACDE",
            "msa": ">q\nAAAA\n",
        }],
    }))
    with pytest.raises(ValueError, match="does not match"):
        run_prepared_workflow(source, tmp_path / "out", run_data_pipeline=True,
                              run_inference=False, write_input_json=False)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("write_input_json", ["false", 1, None])
def test_rejects_non_boolean_snapshot_flag(write_input_json, tmp_path):
    with pytest.raises(ValueError, match="write_input_json must be a boolean"):
        build_workflow_plan(
            tmp_path / "missing.json",
            tmp_path / "outputs",
            write_input_json=write_input_json,
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        (7, (7,)),
        ("7, 9", (7, 9)),
        ([9, 7], (9, 7)),
    ],
)
def test_normalize_seeds_preserves_explicit_order(value, expected):
    assert normalize_seeds(value) == expected


@pytest.mark.parametrize("value", ["1,,2", [1, 1], [-1], [2**32], [True]])
def test_normalize_seeds_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        normalize_seeds(value)


def test_normalize_seeds_generates_one_concrete_uint32(monkeypatch):
    monkeypatch.setattr(
        "esm.esmfold2_wrapper.workflow.secrets.randbelow", lambda limit: limit - 1
    )
    assert normalize_seeds(None) == (2**32 - 1,)


class FakeComplex:
    def __init__(self, seed, sample):
        self.plddt = np.array([0.8], dtype=np.float32)
        self.metadata = SimpleNamespace(chain_lookup={0: "A"})
        self._label = f"data_seed_{seed}_sample_{sample}\n#\n"

    def to_mmcif(self):
        return self._label + "_atom_site.Cartn_x\nATOM 0.0\n#\n"


def fake_result(seed, sample):
    return SimpleNamespace(
        complex=FakeComplex(seed, sample),
        plddt=np.array([0.8], dtype=np.float32),
        ptm=0.7,
        iptm=0.0,
        pae=np.array([[1.0]], dtype=np.float32),
        pde=np.array([[2.0]], dtype=np.float32),
        pair_chains_iptm=np.array([[0.7]], dtype=np.float32),
        output_embedding_pair_pooled=None,
        output_embedding_sequence=None,
        residue_index=np.array([0]),
        entity_id=np.array([0]),
    )


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_workflow_loads_once_for_many_seeds_and_reruns_only_incomplete(
    tmp_path, monkeypatch, damage
):
    source = tmp_path / "target_data.json"
    write_input(source)
    output_dir = tmp_path / "outputs"
    load_calls = []
    fold_seeds = []
    model = SimpleNamespace(
        config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=False)),
        msa_encoder=None,
    )

    def model_loader(*args, **kwargs):
        load_calls.append((args, kwargs))
        return model

    class Builder:
        def fold(self, loaded_model, structure_input, **kwargs):
            assert loaded_model is model
            seed = kwargs["seed"]
            fold_seeds.append(seed)
            return [
                fake_result(seed, sample)
                for sample in range(kwargs["num_diffusion_samples"])
            ]

    monkeypatch.setattr(inference_module, "load_esmfold2_model", model_loader)
    monkeypatch.setattr(inference_module, "_new_input_builder", Builder)

    first = run_prepared_workflow(
        source,
        output_dir,
        write_input_json=False,
        seeds=[7, 9],
        num_diffusion_samples=2,
    )

    assert first.seeds == (7, 9)
    assert len(first.prediction_paths) == 4
    assert len(load_calls) == 1
    assert fold_seeds == [7, 9]

    preserved = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (output_dir / "target").rglob("*")
        if path.is_file() and "seed-7_" in path.name
    }

    monkeypatch.setattr(
        inference_module,
        "load_esmfold2_model",
        lambda *args, **kwargs: pytest.fail("model must not load"),
    )
    monkeypatch.setattr(
        inference_module,
        "_new_input_builder",
        lambda: pytest.fail("builder must not load"),
    )
    skipped = run_prepared_workflow(
        source,
        output_dir,
        write_input_json=False,
        seeds=[7, 9],
        num_diffusion_samples=2,
        skip=True,
        num_sampling_steps=17,
        num_loops=3,
    )
    assert skipped.prediction_paths == first.prediction_paths

    damaged = expected_seed_samples(
        output_dir / "target", seed=9, sample_count=2
    )[1].pde_path
    if damage == "missing":
        damaged.unlink()
    else:
        damaged.write_bytes(b"")
    rerun_folds = []

    class RerunBuilder(Builder):
        def fold(self, loaded_model, structure_input, **kwargs):
            rerun_folds.append((kwargs["seed"], kwargs["num_diffusion_samples"]))
            return super().fold(loaded_model, structure_input, **kwargs)

    monkeypatch.setattr(inference_module, "load_esmfold2_model", model_loader)
    monkeypatch.setattr(inference_module, "_new_input_builder", RerunBuilder)

    run_prepared_workflow(
        source,
        output_dir,
        write_input_json=False,
        seeds=[7, 9],
        num_diffusion_samples=2,
        skip=True,
    )
    assert rerun_folds == [(9, 2)]
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in preserved} == preserved
    assert damaged.stat().st_size > 0


def test_invalid_inference_options_fail_before_data_is_written(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)
    output_dir = tmp_path / "outputs"

    with pytest.raises(ValueError, match="num_diffusion_samples"):
        run_prepared_workflow(
            source,
            output_dir,
            write_input_json=True,
            seeds=1,
            num_diffusion_samples=0,
        )

    assert not output_dir.exists()


def test_combined_workflow_covers_bundle_adapter_and_publication(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({
        "version": 1,
        "name": "paired_target",
        "sequences": [{
            "type": "protein",
            "id": "A",
            "sequence": "ACDE",
            "pairedMsa": ">q\nACDE\n>p\nAC-E\n",
            "unpairedMsa": ">q\nACDE\n>u\nACdDE\n",
        }],
    }))
    output_dir = tmp_path / "outputs"
    model = SimpleNamespace(
        config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=True)),
        msa_encoder=object(),
    )
    load_calls = []
    fold_calls = []

    def model_loader(*args, **kwargs):
        load_calls.append((args, kwargs))
        return model

    class Builder:
        def fold(self, loaded_model, structure_input, **kwargs):
            msa = structure_input.sequences[0].msa
            assert msa.headers == [
                "paired_row_0 key=0",
                "paired_row_1 key=1",
                "unpaired_row_1 key=-1",
            ]
            fold_calls.append(kwargs["seed"])
            return [
                fake_result(kwargs["seed"], sample)
                for sample in range(kwargs["num_diffusion_samples"])
            ]

    monkeypatch.setattr(inference_module, "load_esmfold2_model", model_loader)
    monkeypatch.setattr(inference_module, "_new_input_builder", Builder)

    result = run_prepared_workflow(
        source,
        output_dir,
        write_input_json=True,
        seeds=[11, 13],
    )

    assert result.prepared_path == (
        output_dir / "paired_target/paired_target_data.json"
    ).resolve()
    prepared = json.loads(result.prepared_path.read_text())
    assert prepared["sequences"][0]["pairedMsaPath"] == (
        "msas/paired_target__A_pairedmsa.a3m.zst"
    )
    assert prepared["sequences"][0]["unpairedMsaPath"] == (
        "msas/paired_target__A_unpairedmsa.a3m.zst"
    )
    assert len(load_calls) == 1
    assert fold_calls == [11, 13]
    assert all(path.is_file() for path in result.prediction_paths)


def test_saved_snapshot_survives_manifest_rename_and_cwd_change(
    tmp_path, monkeypatch
):
    source = tmp_path / "input" / "request.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "portable_target",
                "sequences": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "ACDE",
                        "msa": ">q\nACDE\n",
                    }
                ],
            }
        )
    )
    source_bytes = source.read_bytes()
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    model = SimpleNamespace(
        config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=True)),
        msa_encoder=object(),
    )

    class Builder:
        def fold(self, loaded_model, structure_input, **kwargs):
            assert loaded_model is model
            assert structure_input.sequences[0].msa.headers == ["q"]
            return [fake_result(kwargs["seed"], 0)]

    monkeypatch.setattr(inference_module, "load_esmfold2_model", lambda *a, **k: model)
    monkeypatch.setattr(inference_module, "_new_input_builder", Builder)
    data_result = run_prepared_workflow(
        source,
        tmp_path / "prepared",
        write_input_json=True,
        seeds=7,
        num_diffusion_samples=1,
    )
    renamed = data_result.prepared_path.with_name("renamed.json")
    data_result.prepared_path.rename(renamed)
    previous_cwd = os.getcwd()
    try:
        os.chdir(other_cwd)
        inferred = run_prepared_workflow(
            renamed,
            tmp_path / "predictions",
            write_input_json=False,
            seeds=7,
            num_diffusion_samples=1,
        )
    finally:
        os.chdir(previous_cwd)

    assert inferred.prepared_path == renamed.resolve()
    assert inferred.prediction_paths == (
        (
            tmp_path
            / "predictions/portable_target/models/seed-7_sample-0_model.cif"
        ).resolve(),
    )
    assert json.loads(renamed.read_text())["name"] == "portable_target"
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize("mode", ["native", "split"])
@pytest.mark.parametrize("write_snapshot", [False, True])
def test_prediction_reads_replaced_msa_without_reusing_old_snapshot(
    tmp_path, monkeypatch, mode, write_snapshot
):
    source = tmp_path / "input.json"
    msa = tmp_path / "external.a3m"
    first_text = ">query\nACDE\n>original key=42\nACdDE\n"
    next_text = ">query\nACDE\n>replacement key=73\nA-DE\n"
    msa.write_text(first_text)
    fields = {"msaPath": msa.name} if mode == "native" else {
        "pairedMsa": ">query\nACDE\n>padding\n----\n>hit\nAC-E\n",
        "unpairedMsaPath": msa.name,
    }
    source.write_text(json.dumps({
        "version": 1, "name": "target",
        "sequences": [{"type": "protein", "id": ["A", "B"],
                       "sequence": "ACDE", **fields}],
    }))
    source_bytes = source.read_bytes()
    observed = []
    model = SimpleNamespace(
        config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=True)),
        msa_encoder=object(),
    )

    class Builder:
        def fold(self, loaded_model, structure_input, **kwargs):
            observed.append(structure_input.sequences[0].msa)
            return [fake_result(kwargs["seed"], 0)]

    monkeypatch.setattr(inference_module, "load_esmfold2_model", lambda *a, **k: model)
    monkeypatch.setattr(inference_module, "_new_input_builder", Builder)
    output = tmp_path / "out"
    snapshot = output / "target/target_data.json"
    # Publish once, then change the external input at exactly the same path.
    run_prepared_workflow(source, output, write_input_json=True, seeds=7,
                          num_diffusion_samples=1)
    before = {p.relative_to(snapshot.parent): p.read_bytes()
              for p in [snapshot, *snapshot.parent.glob("msas/*")]}
    msa.write_text(next_text)
    result = run_prepared_workflow(
        source, output, write_input_json=write_snapshot, seeds=7,
        num_diffusion_samples=1, run_data_pipeline=False, run_inference=True,
    )
    assert len(observed) == 2
    assert observed[1].headers == (
        ["query", "replacement key=73"] if mode == "native" else
        ["paired_row_0 key=0", "paired_row_2 key=2", "unpaired_row_1 key=-1"]
    )
    assert observed[1].sequences[-1] == "A-DE"
    assert observed[0].sequences[-1] == "ACDE"
    assert source.read_bytes() == source_bytes
    assert result.prepared_path == (snapshot if write_snapshot else source).resolve()
    if write_snapshot:
        entity = json.loads(snapshot.read_text())["sequences"][0]
        field = "msaPath" if mode == "native" else "unpairedMsaPath"
        assert entity[field].startswith("msas/")
        assert read_text_auto(snapshot.parent / entity[field]) == next_text
        if mode == "native":
            assert "pairedMsaPath" not in entity
        else:
            assert "msaPath" not in entity
            assert read_text_auto(snapshot.parent / entity["pairedMsaPath"]) == fields["pairedMsa"]
    else:
        assert {p.relative_to(snapshot.parent): p.read_bytes()
                for p in [snapshot, *snapshot.parent.glob("msas/*")]} == before


def test_default_prediction_does_not_publish_json_or_msa_resources(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    write_input(source)
    model = SimpleNamespace(config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=False)))

    class Builder:
        def fold(self, loaded_model, structure_input, **kwargs):
            return [fake_result(kwargs["seed"], 0)]

    monkeypatch.setattr(inference_module, "load_esmfold2_model", lambda *a, **k: model)
    monkeypatch.setattr(inference_module, "_new_input_builder", Builder)
    result = run_prepared_workflow(source, tmp_path / "out", seeds=7, num_diffusion_samples=1)
    assert result.prepared_path == source.resolve()
    assert result.prediction_paths[0].is_file()
    assert not (tmp_path / "out/target/target_data.json").exists()
    assert not (tmp_path / "out/target/msas").exists()


def test_write_true_refreshes_existing_snapshot_even_when_every_seed_skips(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    write_input(source)
    output = tmp_path / "out"
    sample = expected_seed_samples(output / "target", seed=7, sample_count=1)[0]
    for path in (sample.model_path, sample.summary_path, sample.plddt_path,
                 sample.pae_path, sample.pde_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("nonempty, deliberately not parseable")
    snapshot = output / "target/target_data.json"
    snapshot.write_text("old snapshot to replace")
    monkeypatch.setattr(inference_module, "load_esmfold2_model",
                        lambda *a, **k: pytest.fail("complete seed must skip"))
    result = run_prepared_workflow(source, output, seeds=7, num_diffusion_samples=1,
                                  write_input_json=True, skip=True)
    assert json.loads(snapshot.read_text())["sequences"][0]["sequence"] == "ACDE"
    assert result.prepared_path == snapshot.resolve()
    assert sample.model_path.read_text() == "nonempty, deliberately not parseable"
