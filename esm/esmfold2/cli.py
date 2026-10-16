"""Command-line entry point for ESMFold2 prediction."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from esm.esmfold2.inference import DEFAULT_CHECKPOINT
from esm.esmfold2.workflow import run_prepared_workflow

_CUDA_DEVICE = re.compile(r"^cuda(?::[0-9]+)?$")


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def _device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu"} or _CUDA_DEVICE.fullmatch(normalized):
        return normalized
    raise argparse.ArgumentTypeError(
        "must be auto, cpu, cuda, or a device such as cuda:0"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without importing Torch or model implementations."""
    parser = argparse.ArgumentParser(prog="esmfold2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser(
        "predict",
        help="Validate existing input and/or run ESMFold2, optionally saving input.",
        description=(
            "Run ESMFold2 prediction. This command "
            "does not search for MSAs; omit MSA fields for query-only inference "
            "or provide native/split MSA input in the JSON."
        ),
    )
    predict.add_argument("-y", "--config", type=Path, help="Flat inference YAML; present keys override CLI values.")
    predict.add_argument("-i", "--input", type=Path, required=True)
    predict.add_argument("-o", "--output-dir", type=Path, required=True)
    predict.add_argument(
        "-D", "--run-data-pipeline", type=_boolean, default=True, metavar="BOOL",
        help="Validate and organize existing inputs; no MSA search (default: true).",
    )
    predict.add_argument(
        "-P", "--run-inference", type=_boolean, default=True, metavar="BOOL"
    )
    predict.add_argument(
        "-J", "--write-input-json", "--write_input_json",
        type=_boolean, default=True, metavar="BOOL",
        help="Write/update portable <name>_data.json and MSA resources (default: true).",
    )
    predict.add_argument("-z", "--compress-fold-input", "--compress_fold_input",
                         type=_boolean, default=False, metavar="BOOL")
    predict.add_argument("-f", "--compress-full-confidence", "--compress_full_confidence",
                         type=_boolean, default=False, metavar="BOOL")
    predict.add_argument(
        "-r", "--seeds", "--model-seeds", help="One uint32 seed or comma list."
    )
    predict.add_argument(
        "-n", "--diffusion-samples", type=int, default=5, metavar="N"
    )
    predict.add_argument("-c", "--loops", type=int, default=20, metavar="N")
    predict.add_argument(
        "-p", "--sampling-steps", type=int, default=200, metavar="N"
    )
    predict.add_argument("-k", "--checkpoint", default=DEFAULT_CHECKPOINT)
    predict.add_argument("--esmc-checkpoint", type=Path)
    predict.add_argument("--device", type=_device, default="auto")
    predict.add_argument(
        "--dtype",
        choices=("auto", "fp32"),
        default="auto",
        help="Trunk weight dtype; auto preserves checkpoint precision.",
    )
    predict.add_argument(
        "--esmc-precision", choices=("fp32", "bf16", "fp8"), default="bf16"
    )
    predict.add_argument(
        "--kernel-backend",
        choices=("auto", "fused", "cuequivariance"),
        default="auto",
    )
    predict.add_argument("--cache-dir", type=Path)
    predict.add_argument("--lm-dropout", type=float, default=0.3)
    predict.add_argument("--msa-max-depth", type=int, default=1024)
    predict.add_argument("--msa-column-mask-rate", type=float, default=0.1)
    predict.add_argument(
        "-E", "--include-embeddings", type=_boolean, default=False, metavar="BOOL"
    )
    predict.add_argument(
        "-S", "--skip", type=_boolean, default=False, metavar="BOOL"
    )
    return parser


def _predict(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        result = run_prepared_workflow(
            args.input,
            args.output_dir,
            run_data_pipeline=args.run_data_pipeline,
            run_inference=args.run_inference,
            write_input_json=args.write_input_json,
            compress_fold_input=args.compress_fold_input,
            compress_full_confidence=args.compress_full_confidence,
            seeds=args.seeds,
            skip=args.skip,
            checkpoint=args.checkpoint,
            esmc_checkpoint=args.esmc_checkpoint,
            device=args.device,
            dtype=None if args.dtype == "auto" else args.dtype,
            esmc_precision=args.esmc_precision,
            kernel_backend=(
                None if args.kernel_backend == "auto" else args.kernel_backend
            ),
            cache_dir=args.cache_dir,
            num_loops=args.loops,
            num_sampling_steps=args.sampling_steps,
            num_diffusion_samples=args.diffusion_samples,
            lm_dropout=args.lm_dropout,
            msa_max_depth=args.msa_max_depth,
            msa_column_mask_rate=args.msa_column_mask_rate,
            include_embeddings=args.include_embeddings,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    label = "Input snapshot written" if args.write_input_json else "Input used"
    print(f"{label}: {result.prepared_path}")
    if result.seeds:
        print(f"Seeds: {','.join(map(str, result.seeds))}")
        print(f"Published models: {len(result.prediction_paths)}")
    return 0


def override_flags_with_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Apply only explicitly supplied YAML keys, using the CLI's converters."""
    if args.config is None:
        return
    import yaml

    names = {
        "write_input_json": "write_input_json", "compress_fold_input": "compress_fold_input",
        "compress_full_confidence": "compress_full_confidence", "skip": "skip",
        "num_diffusion_samples": "diffusion_samples", "num_diffusion_steps": "sampling_steps",
        "num_loops": "loops", "checkpoint": "checkpoint", "esmc_checkpoint": "esmc_checkpoint",
        "dtype": "dtype", "esmc_precision": "esmc_precision", "kernel_backend": "kernel_backend",
        "include_embeddings": "include_embeddings",
    }
    predict = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices["predict"]
    actions = {a.dest: a for a in predict._actions}
    try:
        with args.config.open() as handle:
            config = yaml.safe_load(handle)
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("config must be a flat YAML mapping")
        for key, value in config.items():
            if key not in names:
                raise ValueError(f"unsupported config key: {key}")
            action = actions[names[key]]
            expected = bool if action.type is _boolean else int if action.type is int else str
            if type(value) is not expected:
                raise ValueError(f"config {key} must be {expected.__name__}")
            if key in {"checkpoint", "esmc_checkpoint"}:
                path = Path(value).expanduser()
                # Bare org/model identifiers remain Hugging Face IDs. Use ./ to
                # disambiguate a local path with that same spelling.
                is_hub_id = (key == "checkpoint" and re.fullmatch(r"[\w.-]+/[\w.-]+", value)
                             and not value.startswith(".") and not (args.config.parent / path).exists())
                if not is_hub_id:
                    value = str((args.config.resolve().parent / path).resolve())
            converted = action.type(str(value)) if action.type else value
            if action.choices and converted not in action.choices:
                raise ValueError(f"config {key}: invalid choice {value!r}; choose from {action.choices}")
            setattr(args, action.dest, converted)
            print(f"Config override: {key}={converted}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and execute the selected prediction command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    override_flags_with_config(args, parser)
    if args.command == "predict":
        return _predict(args, parser)
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
