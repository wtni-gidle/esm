"""Tests for ESMFold2 prediction publication and resume checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from esm.esmfold2_wrapper.outputs import (
    PredictionOutputError,
    expected_embedding_path,
    expected_seed_samples,
    publish_inference_results,
    seed_outputs_complete,
)


class FakeComplex:
    def __init__(self, sample, chain_ids=("A", "B")):
        self.sample = sample
        self.plddt = np.array([0.7, 0.9], dtype=np.float32)
        self.metadata = SimpleNamespace(
            chain_lookup={index: value for index, value in enumerate(chain_ids)}
        )

    def to_mmcif(self):
        return (
            f"data_sample_{self.sample}\n"
            "_atom_site.Cartn_x\n"
            "ATOM 0.0\n"
            "#\n"
        )


def fake_result(sample, *, embeddings=False):
    return SimpleNamespace(
        complex=FakeComplex(sample),
        plddt=np.array([0.6, 0.7, 0.8], dtype=np.float32),
        ptm=0.71 + sample / 100,
        iptm=0.63 + sample / 100,
        pae=np.full((3, 3), sample + 1.0, dtype=np.float32),
        pde=np.full((3, 3), sample + 2.0, dtype=np.float32),
        pair_chains_iptm=np.array([[0.8, 0.6], [0.7, 0.9]]),
        output_embedding_pair_pooled=(
            np.full((3, 4), sample + 0.5, dtype=np.float32)
            if embeddings
            else None
        ),
        output_embedding_sequence=None,
        residue_index=np.arange(3),
        entity_id=np.array([0, 0, 1]),
    )


def test_expected_paths_use_zero_based_seed_sample_layout(tmp_path):
    expected = expected_seed_samples(tmp_path, seed=17, sample_count=2)

    assert expected[0].model_path == (
        tmp_path / "models/seed-17_sample-0_model.cif"
    ).resolve()
    assert expected[1].summary_path.name == (
        "seed-17_sample-1_summary_confidences.json"
    )
    assert expected[1].plddt_path.name == "plddt_seed-17_sample-1.npz"
    assert expected[1].pae_path.name == "pae_seed-17_sample-1.npz"
    assert expected[1].pde_path.name == "pde_seed-17_sample-1.npz"


def test_publish_preserves_sample_order_and_writes_native_confidence(tmp_path):
    published = publish_inference_results(
        [fake_result(0), fake_result(1)],
        predictions_dir=tmp_path,
        seed=17,
    )

    assert [path.sample for path in published] == [0, 1]
    assert published[0].model_path.read_text().startswith("data_sample_0\n")
    summary = json.loads(published[1].summary_path.read_text())
    assert summary == {
        "seed": 17,
        "sample": 1,
        "mean_plddt": pytest.approx(0.8),
        "ptm": pytest.approx(0.72),
        "iptm": pytest.approx(0.64),
        "chain_ids": ["A", "B"],
        "pair_chains_iptm": {
            "A": {"A": 0.8, "B": 0.6},
            "B": {"A": 0.7, "B": 0.9},
        },
    }
    with np.load(published[0].plddt_path) as archive:
        assert set(archive.files) == {"plddt", "structure_token_plddt"}
        np.testing.assert_allclose(archive["plddt"], [0.6, 0.7, 0.8])
        np.testing.assert_allclose(
            archive["structure_token_plddt"], [0.7, 0.9]
        )
    with np.load(published[1].pde_path) as archive:
        np.testing.assert_array_equal(archive["pde"], np.full((3, 3), 3.0))
    assert seed_outputs_complete(tmp_path, seed=17, sample_count=2)


def test_embeddings_are_seed_level_and_part_of_optional_completeness(tmp_path):
    publish_inference_results(
        [fake_result(0, embeddings=True), fake_result(1, embeddings=True)],
        predictions_dir=tmp_path,
        seed=9,
        include_embeddings=True,
    )

    embedding_path = expected_embedding_path(tmp_path, seed=9)
    with np.load(embedding_path) as archive:
        assert set(archive.files) == {
            "pair_pooled",
            "residue_index",
            "entity_id",
        }
        np.testing.assert_array_equal(archive["pair_pooled"], np.full((3, 4), 0.5))
    assert seed_outputs_complete(
        tmp_path, seed=9, sample_count=2, include_embeddings=True
    )
    embedding_path.unlink()
    assert seed_outputs_complete(tmp_path, seed=9, sample_count=2)
    assert not seed_outputs_complete(
        tmp_path, seed=9, sample_count=2, include_embeddings=True
    )


def test_non_embedding_rerun_leaves_optional_embedding_untouched(tmp_path):
    publish_inference_results(
        [fake_result(0, embeddings=True)],
        predictions_dir=tmp_path,
        seed=11,
        include_embeddings=True,
    )
    embedding_path = expected_embedding_path(tmp_path, seed=11)
    assert embedding_path.is_file()

    publish_inference_results(
        [fake_result(1)], predictions_dir=tmp_path, seed=11
    )

    assert embedding_path.is_file()
    assert seed_outputs_complete(
        tmp_path, seed=11, sample_count=1, include_embeddings=True
    )


def test_skip_checks_nonempty_files_without_parsing_outputs(tmp_path):
    published = publish_inference_results(
        [fake_result(0)], predictions_dir=tmp_path, seed=3
    )
    published[0].summary_path.write_text("not json")
    published[0].model_path.write_text("not cif")
    published[0].pae_path.write_text("not npz")

    assert seed_outputs_complete(tmp_path, seed=3, sample_count=1)

    publish_inference_results([fake_result(0)], predictions_dir=tmp_path, seed=3)
    published[0].pde_path.unlink()
    assert not seed_outputs_complete(tmp_path, seed=3, sample_count=1)


@pytest.mark.parametrize("artifact", ["model", "summary", "plddt", "pae", "pde", "embeddings"])
@pytest.mark.parametrize("state", ["empty", "missing", "directory"])
def test_skip_requires_each_requested_artifact_to_be_a_nonempty_file(
    tmp_path, artifact, state
):
    paths = {}
    for seed in (7, 9):
        for sample in range(2):
            prefix = f"seed-{seed}_sample-{sample}"
            paths.update({
                (seed, sample, "model"): tmp_path / "models" / f"{prefix}_model.cif",
                (seed, sample, "summary"): tmp_path / "summary_confidences" / f"{prefix}_summary_confidences.json",
                **{(seed, sample, name): tmp_path / "full_data" / f"{name}_{prefix}.npz" for name in ("plddt", "pae", "pde")},
            })
        paths[seed, 1, "embeddings"] = tmp_path / "embeddings" / f"seed-{seed}_embeddings.npz"
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"nonempty garbage accepted without parsing")
    assert seed_outputs_complete(tmp_path, seed=9, sample_count=2, include_embeddings=True)
    damaged = paths[9, 1, artifact]
    damaged.unlink()
    if state == "empty":
        damaged.touch()
    elif state == "directory":
        damaged.mkdir()
    assert seed_outputs_complete(tmp_path, seed=7, sample_count=2, include_embeddings=True)
    assert not seed_outputs_complete(tmp_path, seed=9, sample_count=2, include_embeddings=True)
    if artifact == "embeddings":
        assert seed_outputs_complete(tmp_path, seed=9, sample_count=2)


def test_all_results_validate_before_existing_seed_is_replaced(tmp_path):
    old_paths = publish_inference_results(
        [fake_result(0), fake_result(1)], predictions_dir=tmp_path, seed=5
    )
    before = {path: path.read_bytes() for item in old_paths for path in (
        item.model_path,
        item.summary_path,
        item.plddt_path,
        item.pae_path,
        item.pde_path,
    )}
    invalid = fake_result(8)
    invalid.pde = None

    with pytest.raises(PredictionOutputError, match="missing required pde"):
        publish_inference_results(
            [fake_result(7), invalid], predictions_dir=tmp_path, seed=5
        )

    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_invalid_seed_is_rejected(seed, tmp_path):
    with pytest.raises(PredictionOutputError, match="uint32"):
        expected_seed_samples(tmp_path, seed=seed, sample_count=1)
