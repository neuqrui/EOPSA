#!/usr/bin/env bash
# Rubric token rule extraction — per-sample LLM judge + one lexicon audit
#
#   CUDA_VISIBLE_DEVICES=0 bash run_extract_family_token_rules.sh
#   STEPS=judge,aggregate,refine bash run_extract_family_token_rules.sh
#   STEPS=refine TAG=200all bash run_extract_family_token_rules.sh
#   JUDGE_MODE=heuristic N_SAMPLES=32 bash run_extract_family_token_rules.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFETY_RL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd "${SAFETY_RL_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}:${SAFETY_RL_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-${SAFETY_RL_DIR}/datasets/safety_ds_safechain_dsr100_h4400_b2200/train.jsonl}"
TAG="${TAG:-qwen3-1.7b}"
N_SAMPLES="${N_SAMPLES:-200}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TOP_KL="${TOP_KL:-16}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
REFINE_MAX_TOKENS="${REFINE_MAX_TOKENS:-4096}"
STEPS="${STEPS:-all}"
JUDGE_MODE="${JUDGE_MODE:-llm}"
PYTHON="${PYTHON:-python3}"

ARGS=(
  --model_path "${MODEL_PATH}"
  --data "${DATA}"
  --n_samples "${N_SAMPLES}"
  --max_tokens "${MAX_TOKENS}"
  --top_kl_per_sample "${TOP_KL}"
  --tag "${TAG}"
  --judge_mode "${JUDGE_MODE}"
  --judge_model "${JUDGE_MODEL}"
  --judge_max_tokens "${JUDGE_MAX_TOKENS}"
  --judge_concurrency "${JUDGE_CONCURRENCY}"
  --refine_max_tokens "${REFINE_MAX_TOKENS}"
  --steps "${STEPS}"
)

[[ "${FORCE_ROLLOUT:-0}" == "1" ]] && ARGS+=(--force_rollout)
[[ "${FORCE_REFINE:-0}" == "1" ]] && ARGS+=(--force_refine)

echo "[run] MODEL=${MODEL_PATH} N=${N_SAMPLES} rollout_max=${MAX_TOKENS} judge=${JUDGE_MODEL} steps=${STEPS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" "${SCRIPT_DIR}/extract_family_token_rules.py" "${ARGS[@]}"
