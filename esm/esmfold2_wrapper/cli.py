"""Command-line entry point for the first-layer ESMFold2 wrapper."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from esm.esmfold2_wrapper.inference import DEFAULT_CHECKPOINT
from esm.esmfold2_wrapper.workflow import run_prepared_workflow

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
    parser = argparse.ArgumentParser(prog="esmfold2-wrapper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser(
        "predict",
        help="Prepare an input bundle and/or run local ESMFold2 inference.",
        description=(
            "Run the EnsembleFold-compatible ESMFold2 wrapper. This wrapper "
            "does not search for MSAs; omit MSA fields for query-only inference "
            "or provide native/split MSA input in the JSON."
        ),
    )
    predict.add_argument("-i", "--input", type=Path, required=True)
    predict.add_argument("-o", "--output-dir", type=Path, required=True)
    predict.add_argument(
        "-D", "--run-data-pipeline", type=_boolean, default=True, metavar="BOOL"
    )
    predict.add_argument(
        "-P", "--run-inference", type=_boolean, default=True, metavar="BOOL"
    )
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

    print(f"Prepared input: {result.prepared_path}")
    if result.seeds:
        print(f"Seeds: {','.join(map(str, result.seeds))}")
        print(f"Published models: {len(result.prediction_paths)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and execute the selected wrapper command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        return _predict(args, parser)
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
