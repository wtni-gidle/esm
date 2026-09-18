"""Tests for side-effect-free ESMFold2 wrapper planning."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import esm.esmfold2_wrapper.inference as inference_module
from esm.esmfold2_wrapper.outputs import expected_seed_samples
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


def test_data_stage_targets_canonical_prepared_manifest(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)

    plan = build_workflow_plan(
        source, tmp_path / "outputs", run_data_pipeline=True, run_inference=False
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
        run_data_pipeline=False,
        run_inference=True,
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
        source, tmp_path / "outputs", run_data_pipeline=False, run_inference=True
    )

    assert plan.prepared_path == source.resolve()


def test_rejects_an_invocation_with_no_stage(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)
    with pytest.raises(ValueError, match="At least one"):
        build_workflow_plan(
            source,
            tmp_path / "outputs",
            run_data_pipeline=False,
            run_inference=False,
        )


@pytest.mark.parametrize(
    "run_data_pipeline,run_inference",
    [("false", True), (False, "true")],
)
def test_rejects_non_boolean_stage_flags(run_data_pipeline, run_inference, tmp_path):
    with pytest.raises(ValueError, match="must be booleans"):
        build_workflow_plan(
            tmp_path / "missing.json",
            tmp_path / "outputs",
            run_data_pipeline=run_data_pipeline,
            run_inference=run_inference,
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


def test_workflow_loads_once_for_many_seeds_and_reruns_only_incomplete(
    tmp_path, monkeypatch
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
        run_data_pipeline=False,
        run_inference=True,
        seeds=[7, 9],
        num_diffusion_samples=2,
    )

    assert first.seeds == (7, 9)
    assert len(first.prediction_paths) == 4
    assert len(load_calls) == 1
    assert fold_seeds == [7, 9]

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
        run_data_pipeline=False,
        run_inference=True,
        seeds=[7, 9],
        num_diffusion_samples=2,
        skip=True,
    )
    assert skipped.prediction_paths == first.prediction_paths

    expected_seed_samples(
        output_dir / "target", seed=9, sample_count=2
    )[1].pde_path.unlink()
    rerun_folds = []

    class RerunBuilder(Builder):
        def fold(self, loaded_model, structure_input, **kwargs):
            rerun_folds.append(kwargs["seed"])
            return super().fold(loaded_model, structure_input, **kwargs)

    monkeypatch.setattr(inference_module, "load_esmfold2_model", model_loader)
    monkeypatch.setattr(inference_module, "_new_input_builder", RerunBuilder)

    run_prepared_workflow(
        source,
        output_dir,
        run_data_pipeline=False,
        run_inference=True,
        seeds=[7, 9],
        num_diffusion_samples=2,
        skip=True,
    )
    assert rerun_folds == [9]


def test_invalid_inference_options_fail_before_data_is_written(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)
    output_dir = tmp_path / "outputs"

    with pytest.raises(ValueError, match="num_diffusion_samples"):
        run_prepared_workflow(
            source,
            output_dir,
            run_data_pipeline=True,
            run_inference=True,
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
        run_data_pipeline=True,
        run_inference=True,
        seeds=[11, 13],
    )

    assert result.prepared_path == (
        output_dir / "paired_target/paired_target_data.json"
    ).resolve()
    prepared = json.loads(result.prepared_path.read_text())
    assert prepared["sequences"][0]["pairedMsaPath"] == (
        "msas/paired_target__A_paired.a3m.zst"
    )
    assert prepared["sequences"][0]["unpairedMsaPath"] == (
        "msas/paired_target__A_unpaired.a3m.zst"
    )
    assert len(load_calls) == 1
    assert fold_calls == [11, 13]
    assert all(path.is_file() for path in result.prediction_paths)
