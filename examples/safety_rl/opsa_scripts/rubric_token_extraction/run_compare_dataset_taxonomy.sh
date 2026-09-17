#!/usr/bin/env bash
# Compare qwen_rubric category token counts: safechain vs star1 vs mix
#
#   bash run_compare_dataset_taxonomy.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_compare_dataset_taxonomy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFETY_RL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd "${SAFETY_RL_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}:${SAFETY_RL_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
N_SAMPLES="${N_SAMPLES:-100}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python3}"

# Low GPU memory (GPUs mostly occupied)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "[run] GPU=${CUDA_VISIBLE_DEVICES} vllm_mem=${VLLM_GPU_MEMORY_UTILIZATION} model=${MODEL_PATH}"
echo "[run] n=${N_SAMPLES} max_tokens=${MAX_TOKENS} datasets=safechain,star1,mix"

"${PYTHON}" "${SCRIPT_DIR}/compare_dataset_taxonomy_counts.py" \
  --model_path "${MODEL_PATH}" \
  --n_samples "${N_SAMPLES}" \
  --max_tokens "${MAX_TOKENS}" \
  --seed "${SEED}" \
  --datasets "safechain,star1,mix" \
  --classifiers "qwen_rubric,qwen_eopsa" \
  --skip_rollout \
  "$@"
