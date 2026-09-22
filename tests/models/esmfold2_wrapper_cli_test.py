"""CLI and shell-contract tests for the ESMFold2 wrapper."""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from esm.esmfold2_wrapper import cli


def write_input(path):
    path.write_text(json.dumps({
        "version": 1,
        "name": "target",
        "sequences": [{"type": "protein", "id": "A", "sequence": "ACDE"}],
    }))


def test_cli_forwards_every_public_option(monkeypatch, tmp_path, capsys):
    source = tmp_path / "input.json"
    write_input(source)
    calls = []

    def run_workflow(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            prepared_path=tmp_path / "out/target/target_data.json",
            seeds=(9, 7),
            prediction_paths=(Path("one.cif"), Path("two.cif")),
        )

    monkeypatch.setattr(cli, "run_prepared_workflow", run_workflow)
    exit_code = cli.main([
        "predict",
        "-i", str(source),
        "-o", str(tmp_path / "out"),
        "-J", "true",
        "-D", "false", "-P", "true",
        "-r", "9,7",
        "-n", "2",
        "-c", "4",
        "-p", "30",
        "-k", "local-checkpoint",
        "--esmc-checkpoint", "local-esmc",
        "--device", "cuda:1",
        "--dtype", "fp32",
        "--esmc-precision", "fp32",
        "--kernel-backend", "fused",
        "--cache-dir", str(tmp_path / "cache"),
        "--lm-dropout", "0",
        "--msa-max-depth", "256",
        "--msa-column-mask-rate", "0.2",
        "-E", "true",
        "-S", "true",
    ])

    assert exit_code == 0
    assert calls[0][0] == (source, tmp_path / "out")
    assert calls[0][1] == {
        "write_input_json": True,
        "run_data_pipeline": False,
        "run_inference": True,
        "seeds": "9,7",
        "skip": True,
        "checkpoint": "local-checkpoint",
        "esmc_checkpoint": Path("local-esmc"),
        "device": "cuda:1",
        "dtype": "fp32",
        "esmc_precision": "fp32",
        "kernel_backend": "fused",
        "cache_dir": tmp_path / "cache",
        "num_loops": 4,
        "num_sampling_steps": 30,
        "num_diffusion_samples": 2,
        "lm_dropout": 0.0,
        "msa_max_depth": 256,
        "msa_column_mask_rate": 0.2,
        "include_embeddings": True,
    }
    assert "Seeds: 9,7" in capsys.readouterr().out


def test_cli_auto_values_map_to_python_none(monkeypatch, tmp_path):
    source = tmp_path / "input.json"
    write_input(source)
    captured = {}

    def run_workflow(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            prepared_path=source,
            seeds=(),
            prediction_paths=(),
        )

    monkeypatch.setattr(cli, "run_prepared_workflow", run_workflow)
    cli.main([
        "predict",
        "-i", str(source),
        "-o", str(tmp_path / "out"),
    ])

    assert captured["dtype"] is None
    assert captured["kernel_backend"] is None
    assert captured["num_diffusion_samples"] == 5
    assert captured["write_input_json"] is False
    assert captured["run_data_pipeline"] is True
    assert captured["run_inference"] is True


def test_all_skipped_cli_writes_requested_snapshot_without_importing_torch(tmp_path):
    source = tmp_path / "input.json"
    output_dir = tmp_path / "output"
    write_input(source)
    arguments = [
        "predict",
        "-i", str(source),
        "-o", str(output_dir),
        "--write_input_json", "true",
        "-r", "7", "-n", "1", "-S", "true",
    ]
    from esm.esmfold2_wrapper.outputs import expected_seed_samples

    sample = expected_seed_samples(output_dir / "target", seed=7, sample_count=1)[0]
    for path in (sample.model_path, sample.summary_path, sample.plddt_path,
                 sample.pae_path, sample.pde_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("nonempty; skip must not parse this")
    program = (
        "import sys; "
        "from esm.esmfold2_wrapper.cli import main; "
        f"code = main({arguments!r}); "
        "assert code == 0; "
        "assert 'torch' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
    )

    prepared = output_dir / "target/target_data.json"
    assert prepared.is_file()
    assert "Input snapshot written:" in completed.stdout


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["-J", "maybe"], "must be true or false"),
        (["--device", "mps"], "must be auto, cpu, cuda"),
        (["--dtype", "bf16"], "invalid choice"),
    ],
)
def test_cli_rejects_invalid_values(arguments, message, tmp_path, capsys):
    source = tmp_path / "input.json"
    write_input(source)
    with pytest.raises(SystemExit) as error:
        cli.main([
            "predict",
            "-i", str(source),
            "-o", str(tmp_path / "out"),
            *arguments,
        ])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_rejects_no_stage_without_traceback(tmp_path, capsys):
    source = tmp_path / "input.json"
    write_input(source)
    with pytest.raises(SystemExit) as error:
        cli.main([
            "predict",
            "-i", str(source),
            "-o", str(tmp_path / "out"),
            "-D", "false", "-P", "false",
        ])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "At least one" in stderr
    assert "Traceback" not in stderr


def test_shell_preserves_argument_boundaries_and_sets_one_visible_gpu(tmp_path):
    source = tmp_path / "input with space.json"
    write_input(source)
    output_dir = tmp_path / "output with space"
    captured_args = tmp_path / "args.txt"
    captured_device = tmp_path / "device.txt"
    fake_python = tmp_path / "fake python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
        'printf "%s" "${CUDA_VISIBLE_DEVICES-}" > "$CAPTURE_DEVICE"\n'
    )
    fake_python.chmod(0o755)
    script = Path(__file__).parents[2] / "run_esmfold2.sh"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_DEVICE": str(captured_device),
    }

    completed = subprocess.run(
        [
            str(script),
            "-i", str(source),
            "-o", str(output_dir),
            "-d", "3",
            "-J", "TRUE",
            "-D", "false", "-P", "true",
            "-r", "7,9",
            "-n", "2",
            "-m", "local-esmc",
            "-S", "true",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert arguments[:3] == ["-m", "esm.esmfold2_wrapper", "predict"]
    assert arguments[arguments.index("--input") + 1] == str(source)
    assert arguments[arguments.index("--output-dir") + 1] == str(output_dir)
    assert arguments[arguments.index("--seeds") + 1] == "7,9"
    assert arguments[arguments.index("--esmc-checkpoint") + 1] == "local-esmc"
    assert arguments[arguments.index("--device") + 1] == "cuda"
    assert arguments[arguments.index("--write-input-json") + 1] == "true"
    assert arguments[arguments.index("--run-data-pipeline") + 1] == "false"
    assert arguments[arguments.index("--run-inference") + 1] == "true"
    assert captured_device.read_text() == "3"
    assert completed.stderr == ""


def test_shell_is_executable_and_help_needs_no_input():
    script = Path(__file__).parents[2] / "run_esmfold2.sh"
    assert os.access(script, os.X_OK)
    completed = subprocess.run(
        [str(script), "-h"], check=True, capture_output=True, text=True
    )
    assert "Usage:" in completed.stdout


def test_shell_rejects_invalid_boolean_before_calling_python(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)
    script = Path(__file__).parents[2] / "run_esmfold2.sh"

    completed = subprocess.run(
        [
            str(script),
            "-i", str(source),
            "-o", str(tmp_path / "output"),
            "-J", "maybe",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "-J must be true or false" in completed.stderr


def test_data_only_shell_runs_without_torch_or_predictions(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)
    script = Path(__file__).parents[2] / "run_esmfold2.sh"
    result = subprocess.run(
        [str(script), "-i", str(source), "-o", str(tmp_path / "out"),
         "-D", "true", "-P", "false", "-J", "true"],
        env={**os.environ, "PYTHON_BIN": sys.executable,
             "PYTHONPATH": str(Path(__file__).parents[2])},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/target/target_data.json").is_file()
    assert not (tmp_path / "out/target/models").exists()


def test_data_only_cli_does_not_import_torch(tmp_path):
    source = tmp_path / "input.json"
    write_input(source)
    args = ["predict", "-i", str(source), "-o", str(tmp_path / "out"),
            "-D", "true", "-P", "false", "-J", "true"]
    program = ("import sys; from esm.esmfold2_wrapper.cli import main; "
               f"assert main({args!r}) == 0; assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])})
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/target/target_data.json").is_file()


def test_console_script_points_at_lightweight_cli():
    root = Path(__file__).parents[2]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert project["scripts"]["esmfold2-wrapper"] == (
        "esm.esmfold2_wrapper.cli:main"
    )
