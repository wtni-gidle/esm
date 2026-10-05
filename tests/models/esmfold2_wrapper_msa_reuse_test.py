"""MSA I/O regressions; model/SPI and native MSA allocation are boundary doubles.

The manifest reader, A3M validation/conversion, compression and bundle writer
are real. No model weights or native molecular features are needed here.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from esm.esmfold2_wrapper import data, inference, msa_adapter, outputs
from esm.esmfold2_wrapper.workflow import run_prepared_workflow


@pytest.fixture
def inference_boundary(monkeypatch, request):
    native = ModuleType("esm.models.esmfold2")
    for name in ("ProteinInput", "RNAInput", "DNAInput", "LigandInput",
                 "Modification", "StructurePredictionInput"):
        setattr(native, name, SimpleNamespace)
    msa_module = ModuleType("esm.utils.msa")
    msa_module.MSA = SimpleNamespace(from_a3m=lambda stream, **kw: stream.read())
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", native)
    monkeypatch.setitem(sys.modules, "esm.utils.msa", msa_module)
    seen = []

    class Builder:
        def fold(self, model, structure_input, **kwargs):
            seen.append([entity.msa for entity in structure_input.sequences])
            return [object()]

    enabled = getattr(request, "param", True)
    model = SimpleNamespace(
        config=SimpleNamespace(msa_encoder=SimpleNamespace(enabled=enabled)),
        msa_encoder=object() if enabled else None,
    )
    monkeypatch.setattr(inference, "load_esmfold2_model", lambda *a, **kw: model)
    monkeypatch.setattr(inference, "_new_input_builder", Builder)
    monkeypatch.setattr(outputs, "publish_inference_results", lambda *a, **kw: [object()])
    return seen


@pytest.mark.parametrize("mode", ["native", "split"])
@pytest.mark.parametrize("inference_boundary", [True, False], indirect=True)
@pytest.mark.parametrize("data_stage,publish", [(True, True), (True, False), (False, False), (False, True)])
def test_msa_is_read_and_wrapper_parsed_once_per_invocation(
    tmp_path, monkeypatch, inference_boundary, mode, data_stage, publish
):
    paired = ">query\nACDE\n>padding\n----\n>hit\nAC-E\n"
    unpaired = ">query\nACDE\n>unpaired\nACdDE\n"
    native = ">query\nACDE\n>hit key=2\nACdDE\n"
    entity = {"type": "protein", "id": "A", "sequence": "ACDE"}
    sources = {}
    for field, text in (([("msaPath", native)]) if mode == "native" else
                        [("pairedMsaPath", paired), ("unpairedMsaPath", unpaired)]):
        path = tmp_path / f"{field}.a3m"
        path.write_text(text)
        entity[field] = path.name
        sources[path.resolve()] = text
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"version": 1, "name": "target", "sequences": [entity]}))
    reads, parses = Counter(), Counter()
    read_original, parse_original = data.read_text_auto, msa_adapter.parse_a3m_records

    def read(path):
        reads[Path(path).resolve()] += 1
        return read_original(path)

    def parse(text, location):
        parses[text] += 1
        return parse_original(text, location)

    monkeypatch.setattr(data, "read_text_auto", read)
    if hasattr(inference, "read_text_auto"):
        monkeypatch.setattr(inference, "read_text_auto", read)
    monkeypatch.setattr(msa_adapter, "parse_a3m_records", parse)
    # Support direct imports without substituting parser behavior.
    if hasattr(data, "parse_a3m_records"):
        monkeypatch.setattr(data, "parse_a3m_records", parse)

    expected = native if mode == "native" else (
        ">paired_row_0 key=0\nACDE\n>paired_row_2 key=2\nAC-E\n"
        ">unpaired_row_1 key=-1\nACdDE\n"
    )
    for iteration in range(2):
        reads.clear()
        parses.clear()
        run_prepared_workflow(manifest, tmp_path / f"out{iteration}",
                              run_data_pipeline=data_stage, write_input_json=publish,
                              compress_fold_input=True, seeds=[7, 9], num_diffusion_samples=1)
        assert inference_boundary[-2:] == [[expected], [expected]]
        assert reads == Counter({path: 1 for path in sources})
        assert parses == Counter({text: 1 for text in sources.values()})
        # A subsequent call must observe new bytes, not a process-wide cache.
        for path, text in list(sources.items()):
            updated = text.replace("ACdDE", "ACddDE")
            path.write_text(updated)
            sources[path] = updated
        expected = expected.replace("ACdDE", "ACddDE")


@pytest.mark.parametrize("data_stage,publish", [(True, True), (True, False), (False, False), (False, True)])
@pytest.mark.parametrize("bad_input,error", [("query", "does not match"), ("depth", "same original row count")])
def test_reuse_keeps_input_checks_before_publication(
    tmp_path, inference_boundary, data_stage, publish, bad_input, error
):
    entities = [
        {"type": "protein", "id": "A", "sequence": "ACDE",
         "pairedMsa": ">q\nACDE\n>hit\nAC-E\n", "unpairedMsa": ">q\nACDE\n"},
        {"type": "protein", "id": "B", "sequence": "FGHI",
         "pairedMsa": ">q\nFGHI\n>hit\nFG-I\n", "unpairedMsa": ">q\nFGHI\n"},
    ]
    if bad_input == "query":
        entities[1]["unpairedMsa"] = ">wrong\nAAAA\n"
    else:
        entities[1]["pairedMsa"] = ">q\nFGHI\n"
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"version": 1, "name": "target", "sequences": entities}))
    with pytest.raises(ValueError, match=error):
        run_prepared_workflow(manifest, tmp_path / "out", run_data_pipeline=data_stage,
                              write_input_json=publish, seeds=1, num_diffusion_samples=1)
    assert not (tmp_path / "out").exists()
    assert inference_boundary == []


def test_direct_conversion_still_requires_resolved_paths(tmp_path):
    from esm.esmfold2_wrapper.input import PreparedInput
    from esm.esmfold2_wrapper.inference import prepared_to_structure_prediction_input

    prepared = PreparedInput.from_dict({"version": 1, "name": "target", "sequences": [
        {"type": "protein", "id": "A", "sequence": "ACDE", "msaPath": "relative.a3m"}
    ]})
    with pytest.raises(ValueError, match="MSA paths must be resolved"):
        prepared_to_structure_prediction_input(prepared)


@pytest.mark.parametrize("inference_boundary", [False], indirect=True)
@pytest.mark.parametrize("mode", ["native", "split"])
def test_encoder_disabled_still_forwards_query_only_msa(
    tmp_path, inference_boundary, mode
):
    fields = {"msa": ">q\nACDE\n"} if mode == "native" else {
        "pairedMsa": "", "unpairedMsa": ">q\nACDE\n",
    }
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"version": 1, "name": "target", "sequences": [
        {"type": "protein", "id": "A", "sequence": "ACDE", **fields},
    ]}))
    run = inference.run_esmfold2_inference(manifest, num_diffusion_samples=1)
    expected = ">q\nACDE\n" if mode == "native" else ">unpaired_row_0 key=-1\nACDE\n"
    assert inference_boundary == [[expected]]
    assert len(run.results) == 1
