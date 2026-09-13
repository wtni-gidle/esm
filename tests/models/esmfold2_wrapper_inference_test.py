"""Tests for prepared-input conversion and the one-seed inference boundary."""

import json
from types import SimpleNamespace

import pytest

import esm.esmfold2_wrapper.inference as inference_module
from esm.esmfold2_wrapper.inference import (
    load_esmfold2_model,
    load_structure_prediction_input,
    model_supports_msa,
    run_esmfold2_inference,
)
from esm.esmfold2_wrapper.input import PreparedInputError


def write_manifest(path, sequences):
    path.write_text(json.dumps({"version": 1, "name": "target", "sequences": sequences}))


def test_manifest_maps_to_native_entities_in_declared_order(tmp_path):
    manifest = tmp_path / "target_data.json"
    write_manifest(
        manifest,
        [
            {
                "type": "protein",
                "id": ["A", "B"],
                "sequence": "ACDE",
                "modifications": [{"position": 1, "ccd": "MSE"}],
            },
            {"type": "rna", "id": "R", "sequence": "ACGU"},
            {"type": "dna", "id": "D", "sequence": "ACGT"},
            {"type": "ligand", "id": "L", "ccd": ["ATP"]},
            {"type": "ligand", "id": "M", "smiles": "[Mg+2]"},
        ],
    )

    _, structure_input = load_structure_prediction_input(manifest)

    assert [type(entity).__name__ for entity in structure_input.sequences] == [
        "ProteinInput",
        "RNAInput",
        "DNAInput",
        "LigandInput",
        "LigandInput",
    ]
    protein = structure_input.sequences[0]
    assert protein.id == ["A", "B"]
    assert protein.msa is None
    assert protein.modifications[0].position == 1
    assert structure_input.sequences[3].ccd == ["ATP"]
    assert structure_input.sequences[4].smiles == "[Mg+2]"


def test_split_components_become_one_native_msa(tmp_path):
    manifest = tmp_path / "target_data.json"
    write_manifest(
        manifest,
        [{
            "type": "protein",
            "id": "A",
            "sequence": "ACDE",
            "pairedMsa": ">q\nACDE\n>p\nAC-E\n",
            "unpairedMsa": ">q\nACDE\n>u\nACdDE\n",
        }],
    )

    _, structure_input = load_structure_prediction_input(manifest)
    msa = structure_input.sequences[0].msa

    assert msa.headers == [
        "paired_row_0 key=0",
        "paired_row_1 key=1",
        "unpaired_row_1 key=-1",
    ]
    assert msa.sequences == ["ACDE", "AC-E", "ACDE"]
    assert msa.deletions[2, 2] == 1


@pytest.mark.parametrize(
    "enabled,disabled,expected",
    [(True, False, True), (False, False, False), (True, True, True)],
)
def test_model_msa_capability_comes_from_config(enabled, disabled, expected):
    model = SimpleNamespace(
        config=SimpleNamespace(
            msa_encoder=SimpleNamespace(enabled=enabled),
            disable_msa_features=disabled,
        ),
        msa_encoder=object() if enabled else None,
    )
    assert model_supports_msa(model) is expected


def test_experimental_checkpoint_is_rejected_before_inference():
    model = SimpleNamespace(config=SimpleNamespace(type="experimental"))
    with pytest.raises(ValueError, match="do not produce the required PDE"):
        inference_module._validate_loaded_model(model)


def test_explicit_local_esmc_checkpoint_is_attached_after_trunk_load(monkeypatch):
    from esm.models.esmfold2 import EsmFold2Model

    calls = []

    class FakeModel:
        config = SimpleNamespace(esmc_config=None)

        def eval(self):
            return self

        def load_esmc(self, checkpoint, *, precision):
            calls.append((checkpoint, precision))

    model = FakeModel()
    from_pretrained_calls = []

    def from_pretrained(*args, **kwargs):
        from_pretrained_calls.append((args, kwargs))
        return model

    monkeypatch.setattr(EsmFold2Model, "from_pretrained", from_pretrained)

    assert load_esmfold2_model(
        "local-fold", device="cpu", esmc_checkpoint="local-esmc"
    ) is model
    assert from_pretrained_calls == [
        (("local-fold",), {
            "device": "cpu",
            "dtype": None,
            "esmc_precision": "bf16",
            "load_esmc": False,
        })
    ]
    assert calls == [("local-esmc", "bf16")]


def test_inference_passes_public_fold_options_and_normalizes_results(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "target_data.json"
    write_manifest(
        manifest,
        [{"type": "protein", "id": "A", "sequence": "ACDE"}],
    )
    model = SimpleNamespace(
        config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=False)),
        msa_encoder=None,
    )
    load_calls = []
    fold_calls = []
    sentinels = [object(), object()]

    def model_loader(checkpoint, **kwargs):
        load_calls.append((checkpoint, kwargs))
        return model

    class Builder:
        def fold(self, loaded_model, structure_input, **kwargs):
            fold_calls.append((loaded_model, structure_input, kwargs))
            return sentinels

    monkeypatch.setattr(inference_module, "load_esmfold2_model", model_loader)
    monkeypatch.setattr(inference_module, "_new_input_builder", Builder)

    run = run_esmfold2_inference(
        manifest,
        checkpoint="local-checkpoint",
        device="cuda",
        num_loops=3,
        num_sampling_steps=40,
        num_diffusion_samples=2,
        seed=7,
        lm_dropout=0,
        msa_max_depth=512,
        msa_column_mask_rate=0.2,
    )

    assert load_calls[0][0] == "local-checkpoint"
    assert load_calls[0][1]["device"] == "cuda"
    assert fold_calls[0][0] is model
    assert fold_calls[0][2] == {
        "num_loops": 3,
        "num_sampling_steps": 40,
        "num_diffusion_samples": 2,
        "seed": 7,
        "lm_dropout": 0,
        "msa_max_depth": 512,
        "msa_column_mask_rate": 0.2,
        "include_embeddings": False,
        "complex_id": "target",
    }
    assert run.results == tuple(sentinels)


def test_fast_like_checkpoint_rejects_declared_msa_before_builder(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "target_data.json"
    write_manifest(
        manifest,
        [{
            "type": "protein",
            "id": "A",
            "sequence": "ACDE",
            "msa": ">q\nACDE\n",
        }],
    )
    model = SimpleNamespace(
        config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=False)),
        msa_encoder=None,
    )
    monkeypatch.setattr(
        inference_module, "load_esmfold2_model", lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        inference_module,
        "_new_input_builder",
        lambda: pytest.fail("builder must not be created"),
    )

    with pytest.raises(ValueError, match="does not support MSA"):
        run_esmfold2_inference(manifest)


def test_inference_only_revalidates_cross_entity_paired_depth(tmp_path):
    manifest = tmp_path / "target_data.json"
    write_manifest(
        manifest,
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
        load_structure_prediction_input(manifest)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"num_loops": 0}, "num_loops"),
        ({"num_diffusion_samples": 0}, "num_diffusion_samples"),
        ({"seed": -1}, "uint32"),
        ({"msa_column_mask_rate": 1.1}, "between 0 and 1"),
        ({"dtype": "bf16"}, "dtype"),
        ({"esmc_precision": "int8"}, "esmc_precision"),
        ({"kernel_backend": "unknown"}, "kernel_backend"),
    ],
)
def test_invalid_fold_options_fail_before_loading_the_manifest(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_esmfold2_inference(tmp_path / "missing.json", **kwargs)


def test_invalid_manifest_fails_before_loading_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference_module,
        "load_esmfold2_model",
        lambda *args, **kwargs: pytest.fail("model must not be loaded"),
    )

    with pytest.raises(PreparedInputError):
        run_esmfold2_inference(tmp_path / "missing.json")
