#!/bin/bash
# ============================================================
# SafeChain 有害子集 → Teacher 推理 → Llama-Guard DSR
# （不评测良性数据）
#
# 有害划分与训练一致（字段 label）:
#   harmful = vanilla_harmful + adversarial_harmful
# ============================================================
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL=${MODEL:-qwen3-1.7b}
HARMFUL_N=${HARMFUL_N:-200}
SEED=${SEED:-42}
ENABLE_THINKING=${ENABLE_THINKING:-true}
USE_RESPONSE_HINT=${USE_RESPONSE_HINT:-false}
SAFECHAIN_LOCAL=${SAFECHAIN_LOCAL:-"${SCRIPT_DIR}/../datasets/SafeChain_raw/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs/safechain_teacher_dsr"}

# 可选：只保留部分 harmful label，例如
#   HARMFUL_LABELS=vanilla_harmful
#   HARMFUL_LABELS=adversarial_harmful
HARMFUL_LABELS=${HARMFUL_LABELS:-}

export GUARD_MODEL_PATH=${GUARD_MODEL_PATH:-"meta-llama/Llama-Guard-3-8B"}
export MODEL_PATH_QWEN3_1_7B=${MODEL_PATH_QWEN3_1_7B:-"Qwen/Qwen3-1.7B"}
export MODEL_PATH_QWEN3_4B=${MODEL_PATH_QWEN3_4B:-"Qwen/Qwen3-4B"}
export MODEL_PATH_DS_1_5B=${MODEL_PATH_DS_1_5B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
export MODEL_PATH_DS_7B=${MODEL_PATH_DS_7B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
export MODELS_TO_RUN="${MODEL}"
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}

ARGS=(
  --model "${MODEL}"
  --harmful "${HARMFUL_N}"
  --seed "${SEED}"
  --safechain_local "${SAFECHAIN_LOCAL}"
  --output_dir "${OUTPUT_DIR}"
  --temperature "${TEACHER_TEMPERATURE:-0.6}"
  --top_p "${TEACHER_TOP_P:-0.95}"
  --max_tokens "${TEACHER_MAX_TOKENS:-4096}"
)
if [ -n "${HARMFUL_LABELS}" ]; then
  ARGS+=(--harmful_labels "${HARMFUL_LABELS}")
fi
if [ "${USE_RESPONSE_HINT}" = "true" ]; then
  ARGS+=(--use_response_hint)
fi
if [ "${ENABLE_THINKING}" != "true" ]; then
  ARGS+=(--no_thinking)
fi

echo "============================================================"
echo "  SafeChain Teacher DSR (harmful only)"
echo "  model=${MODEL}  harmful=${HARMFUL_N}"
echo "  safechain=${SAFECHAIN_LOCAL}"
echo "  harmful_labels=${HARMFUL_LABELS:-ALL_HARMFUL}"
echo "============================================================"

python3 filter_safechain_teacher_dsr.py "${ARGS[@]}"
echo "结果目录: ${OUTPUT_DIR}"
