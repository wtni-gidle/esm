#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 -i <input.json> -o <output_dir> [options]"
    echo ""
    echo "Required:"
    echo "  -i  Input JSON or prepared <name>_data.json."
    echo "  -o  Output root directory."
    echo ""
    echo "Options:"
    echo "  -d  Single visible CUDA device ID. (default: 0)"
    echo "  -D  Run data pipeline: true/false. (default: true)"
    echo "  -P  Run inference: true/false. (default: true)"
    echo "  -r  One seed or comma-separated seeds."
    echo "  -n  Diffusion samples per seed. (default: 5)"
    echo "  -c  ESMFold2 folding loops. (default: 20)"
    echo "  -p  Diffusion sampling steps. (default: 200)"
    echo "  -k  Checkpoint path or Hugging Face ID. (default: biohub/ESMFold2)"
    echo "  -m  Local ESMC checkpoint directory. (default: checkpoint configuration)"
    echo "  -E  Write seed-level embeddings: true/false. (default: false)"
    echo "  -S  Skip seeds whose canonical output files exist. (default: false)"
    echo "  -h  Show this help."
    echo ""
    echo "Examples:"
    echo "  $0 -i target.json -o results -D true -P false"
    echo "  $0 -i results/target/target_data.json -o results -D false -P true -r 1,2,3 -S true"
}

normalize_boolean() {
    local option_name=$1
    local value
    value=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
    case "$value" in
        true|false) printf '%s' "$value" ;;
        *)
            echo "Error: $option_name must be true or false: $2" >&2
            exit 2
            ;;
    esac
}

while getopts "i:o:d:D:P:r:n:c:p:k:m:E:S:h" option; do
    case "$option" in
        i) input_path=$OPTARG ;;
        o) output_dir=$OPTARG ;;
        d) gpu_device=$OPTARG ;;
        D) run_data_pipeline=$OPTARG ;;
        P) run_inference=$OPTARG ;;
        r) model_seeds=$OPTARG ;;
        n) diffusion_samples=$OPTARG ;;
        c) loops=$OPTARG ;;
        p) sampling_steps=$OPTARG ;;
        k) checkpoint=$OPTARG ;;
        m) esmc_checkpoint=$OPTARG ;;
        E) include_embeddings=$OPTARG ;;
        S) skip=$OPTARG ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ "${input_path:-}" == "" || "${output_dir:-}" == "" ]]; then
    usage
    exit 2
fi
if [[ ! -f "$input_path" ]]; then
    echo "Error: input JSON does not exist: $input_path" >&2
    exit 2
fi

run_data_pipeline=${run_data_pipeline:-true}
run_inference=${run_inference:-true}
gpu_device=${gpu_device:-${CUDA_VISIBLE_DEVICES:-0}}
diffusion_samples=${diffusion_samples:-5}
loops=${loops:-20}
sampling_steps=${sampling_steps:-200}
checkpoint=${checkpoint:-biohub/ESMFold2}
include_embeddings=${include_embeddings:-false}
skip=${skip:-false}

run_data_pipeline=$(normalize_boolean -D "$run_data_pipeline")
run_inference=$(normalize_boolean -P "$run_inference")
include_embeddings=$(normalize_boolean -E "$include_embeddings")
skip=$(normalize_boolean -S "$skip")

if [[ "$run_inference" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_device"
fi

python_bin=${PYTHON_BIN:-python}
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Error: Python executable is unavailable: $python_bin" >&2
    exit 2
fi

command_args=(
    -m esm.esmfold2_wrapper predict
    --input "$input_path"
    --output-dir "$output_dir"
    --run-data-pipeline "$run_data_pipeline"
    --run-inference "$run_inference"
    --diffusion-samples "$diffusion_samples"
    --loops "$loops"
    --sampling-steps "$sampling_steps"
    --checkpoint "$checkpoint"
    --include-embeddings "$include_embeddings"
    --skip "$skip"
)
if [[ "${model_seeds:-}" != "" ]]; then
    command_args+=(--seeds "$model_seeds")
fi
if [[ "${esmc_checkpoint:-}" != "" ]]; then
    command_args+=(--esmc-checkpoint "$esmc_checkpoint")
fi
if [[ "$run_inference" == "true" ]]; then
    command_args+=(--device cuda)
fi

echo "$python_bin ${command_args[*]}"
"$python_bin" "${command_args[@]}"
