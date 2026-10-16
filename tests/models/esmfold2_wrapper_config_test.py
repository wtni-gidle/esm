"""Config overrides are applied before the real data-only workflow runs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from esm.esmfold2 import cli


def test_config_false_overrides_cli_in_actual_workflow(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"version": 1, "name": "target", "sequences": [
        {"type": "protein", "id": "A", "sequence": "ACDE"}]}))
    config = tmp_path / "inference.yaml"
    config.write_text("write_input_json: false\n")
    assert cli.main(["predict", "-i", str(source), "-o", str(tmp_path / "out"),
                     "-P", "false", "-J", "true", "--config", str(config)]) == 0
    assert not (tmp_path / "out/target/target_data.json").exists()


def test_config_sampling_precedence_and_paths(tmp_path):
    config = tmp_path / "inference.yaml"
    config.write_text("num_diffusion_samples: 2\nnum_diffusion_steps: 3\nnum_loops: 0\n"
                      "checkpoint: ./weights/fold\nesmc_checkpoint: weights/esmc\n")
    parser = cli.build_parser()
    args = parser.parse_args(["predict", "-i", "in.json", "-o", "out", "-n", "8",
                              "--dtype", "fp32", "--config", str(config)])
    cli.override_flags_with_config(args, parser)
    assert (args.diffusion_samples, args.sampling_steps, args.loops) == (2, 3, 0)
    assert args.dtype == "fp32"
    assert args.checkpoint == str(tmp_path / "weights/fold")
    assert args.esmc_checkpoint == tmp_path / "weights/esmc"
    config.write_text("checkpoint: biohub/ESMFold2\n")
    cli.override_flags_with_config(args, parser)
    assert args.checkpoint == "biohub/ESMFold2"


@pytest.mark.parametrize("setting", ["unknown: 1", "seeds: [1]", "run_inference: false",
    "config: other.yaml", "num_diffusion_samples: true", "num_loops: 2.5",
    "write_input_json: 'false'", "checkpoint: null", "dtype: bf16", "[1, 2]"])
def test_invalid_config_fails_before_workflow(tmp_path, setting, capsys):
    config = tmp_path / "inference.yaml"
    config.write_text(setting)
    with pytest.raises(SystemExit) as exc:
        cli.main(["predict", "-i", "missing.json", "-o", "out", "--config", str(config)])
    assert exc.value.code == 2
    assert "unrecognized arguments" not in capsys.readouterr().err


def test_shell_long_options_forward_config_and_inherit_gpu(tmp_path):
    source = tmp_path / "input file.json"
    source.write_text("{}")
    executable = tmp_path / "capture"
    executable.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                          "print(json.dumps([sys.argv[1:],os.environ['CUDA_VISIBLE_DEVICES']]))\n")
    executable.chmod(0o755)
    shell = Path(__file__).parents[2] / "run_esmfold2.sh"
    result = subprocess.run(["bash", str(shell), "--input", str(source), "--output-dir",
        str(tmp_path / "out"), "--config", "config file.yaml", "--model-seeds", "0,1",
        "--run-data-pipeline", "false", "--run-inference", "true"],
        env={**os.environ, "PYTHON_BIN": str(executable), "CUDA_VISIBLE_DEVICES": "7"},
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args, gpu = json.loads(result.stdout.splitlines()[-1])
    assert args[args.index("--config") + 1] == "config file.yaml"
    assert args[args.index("--seeds") + 1] == "0,1"
    assert gpu == "7"
    assert subprocess.run(["bash", str(shell), "--help"], capture_output=True).returncode == 0


@pytest.mark.parametrize("contents", ["", "# comment only\n", "{}"])
def test_empty_config_keeps_cli_and_defaults(tmp_path, contents):
    config = tmp_path / "inference.yaml"
    config.write_text(contents)
    parser = cli.build_parser()
    args = parser.parse_args(["predict", "-i", "in.json", "-o", "out", "-n", "8", "--config", str(config)])
    cli.override_flags_with_config(args, parser)
    assert args.diffusion_samples == 8
    assert args.loops == 20
