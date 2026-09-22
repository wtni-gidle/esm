"""Lightweight input preparation and ESMFold2 inference with optional snapshots."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from esm.esmfold2_wrapper.input import (
    PreparedInput,
    PreparedInputError,
    load_prepared_input,
    prepared_input_path,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowPlan:
    """Resolved paths and snapshot policy for one prediction invocation."""

    input_path: Path
    output_dir: Path
    name: str
    job_dir: Path
    prepared_path: Path
    predictions_dir: Path
    run_data_pipeline: bool
    run_inference: bool
    write_input_json: bool
    prepared_input: PreparedInput


@dataclass(frozen=True)
class WorkflowResult:
    """Published resources and predictions from one wrapper invocation."""

    prepared_path: Path
    seeds: tuple[int, ...]
    prediction_paths: tuple[Path, ...]


def normalize_seeds(
    seeds: str | int | Sequence[int] | None,
) -> tuple[int, ...]:
    """Normalize one or more unique uint32 seeds in requested order."""
    if seeds is None or (isinstance(seeds, str) and not seeds.strip()):
        generated = secrets.randbelow(2**32)
        logger.info("No seed supplied; generated seed %d", generated)
        return (generated,)
    if isinstance(seeds, str):
        items = seeds.split(",")
        if any(not item.strip() for item in items):
            raise ValueError("seeds must not contain empty items")
        try:
            values = tuple(int(item.strip()) for item in items)
        except ValueError as error:
            raise ValueError(
                "seeds must be a comma-separated list of integers"
            ) from error
    elif type(seeds) is int:
        values = (seeds,)
    else:
        values = tuple(seeds)

    if not values:
        raise ValueError("At least one seed is required for inference")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values):
        raise ValueError("Every seed must be an integer in the uint32 range")
    if len(values) != len(set(values)):
        raise ValueError("Seeds must be unique within one invocation")
    return values


def build_workflow_plan(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    write_input_json: bool = False,
    validate_resources: bool = True,
) -> WorkflowPlan:
    """Validate an invocation without creating directories or loading weights."""
    if type(write_input_json) is not bool:
        raise ValueError("write_input_json must be a boolean")
    if type(run_data_pipeline) is not bool or type(run_inference) is not bool:
        raise ValueError("run_data_pipeline and run_inference must be booleans")
    if not run_data_pipeline and not run_inference:
        raise ValueError("At least one of run_data_pipeline or run_inference must be true.")

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input path does not exist or is not a file: {source}")
    if source.suffix.lower() != ".json":
        raise PreparedInputError("The wrapper expects a JSON input.")

    output_root = Path(output_dir).expanduser().resolve()
    prepared = load_prepared_input(source)
    if validate_resources:
        prepared = prepared.validate_resources(source)
    job_dir = output_root / prepared.name
    manifest = (
        prepared_input_path(output_root, prepared.name)
        if write_input_json
        else source
    )
    return WorkflowPlan(
        input_path=source,
        output_dir=output_root,
        name=prepared.name,
        job_dir=job_dir,
        prepared_path=manifest,
        predictions_dir=job_dir,
        run_data_pipeline=run_data_pipeline,
        run_inference=run_inference,
        write_input_json=write_input_json,
        prepared_input=prepared,
    )


def run_prepared_workflow(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    write_input_json: bool = False,
    seeds: str | int | Sequence[int] | None = None,
    skip: bool = False,
    checkpoint: str | Path = "biohub/ESMFold2",
    esmc_checkpoint: str | Path | None = None,
    device: str = "auto",
    dtype: str | None = None,
    esmc_precision: str = "bf16",
    kernel_backend: str | None = None,
    cache_dir: str | Path | None = None,
    num_loops: int = 20,
    num_sampling_steps: int = 200,
    num_diffusion_samples: int = 5,
    lm_dropout: float | None = 0.3,
    msa_max_depth: int | None = 1024,
    msa_column_mask_rate: float = 0.1,
    include_embeddings: bool = False,
) -> WorkflowResult:
    """Validate existing input and/or predict, independently controlling writes."""
    if type(skip) is not bool:
        raise ValueError("skip must be a boolean")
    if type(include_embeddings) is not bool:
        raise ValueError("include_embeddings must be a boolean")
    plan = build_workflow_plan(
        input_path,
        output_dir,
        run_data_pipeline=run_data_pipeline,
        run_inference=run_inference,
        write_input_json=write_input_json,
    )
    normalized_seeds: tuple[int, ...] = ()
    if plan.run_inference:
        from esm.esmfold2_wrapper.inference import (
            _validate_fold_options,
            _validate_model_options,
        )

        normalized_seeds = normalize_seeds(seeds)
        _validate_model_options(
            device=device,
            dtype=dtype,
            esmc_precision=esmc_precision,
            kernel_backend=kernel_backend,
        )
        _validate_fold_options(
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=normalized_seeds[0],
            lm_dropout=lm_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
        )

    manifest_path = plan.input_path
    if plan.write_input_json:
        from esm.esmfold2_wrapper.data import prepare_data_bundle

        prepare_data_bundle(plan.input_path, plan.prepared_path)
        manifest_path = plan.prepared_path
    elif plan.run_data_pipeline:
        from esm.esmfold2_wrapper.data import validate_data_input

        validate_data_input(plan.input_path)

    if not plan.run_inference:
        return WorkflowResult(prepared_path=manifest_path, seeds=(), prediction_paths=())

    from esm.esmfold2_wrapper.inference import (
        _new_input_builder,
        _run_with_loaded_model,
        load_esmfold2_model,
        load_structure_prediction_input,
    )
    from esm.esmfold2_wrapper.outputs import (
        expected_seed_samples,
        publish_inference_results,
        seed_outputs_complete,
    )

    pending_seeds = tuple(
        seed
        for seed in normalized_seeds
        if not skip
        or not seed_outputs_complete(
            plan.predictions_dir,
            seed=seed,
            sample_count=num_diffusion_samples,
            include_embeddings=include_embeddings,
        )
    )
    skipped = tuple(seed for seed in normalized_seeds if seed not in pending_seeds)
    if skipped:
        logger.info("Skipping complete seeds: %s", ", ".join(map(str, skipped)))

    expected_model_paths = tuple(
        sample.model_path
        for seed in normalized_seeds
        for sample in expected_seed_samples(
            plan.predictions_dir,
            seed=seed,
            sample_count=num_diffusion_samples,
        )
    )
    if not pending_seeds:
        return WorkflowResult(
            prepared_path=manifest_path,
            seeds=normalized_seeds,
            prediction_paths=expected_model_paths,
        )

    prepared, structure_input = load_structure_prediction_input(manifest_path)
    model = load_esmfold2_model(
        checkpoint,
        esmc_checkpoint=esmc_checkpoint,
        device=device,
        dtype=dtype,
        esmc_precision=esmc_precision,
        kernel_backend=kernel_backend,
        cache_dir=cache_dir,
    )
    builder = _new_input_builder()

    for seed in pending_seeds:
        logger.info("Running ESMFold2 seed %d", seed)
        run = _run_with_loaded_model(
            manifest_path,
            prepared=prepared,
            structure_input=structure_input,
            model=model,
            builder=builder,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
            lm_dropout=lm_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            include_embeddings=include_embeddings,
        )
        published = publish_inference_results(
            run.results,
            predictions_dir=plan.predictions_dir,
            seed=seed,
            include_embeddings=include_embeddings,
        )
        if len(published) != num_diffusion_samples:
            raise RuntimeError(
                f"Seed {seed} published {len(published)} samples; "
                f"expected {num_diffusion_samples}"
            )

    return WorkflowResult(
        prepared_path=manifest_path,
        seeds=normalized_seeds,
        prediction_paths=expected_model_paths,
    )
