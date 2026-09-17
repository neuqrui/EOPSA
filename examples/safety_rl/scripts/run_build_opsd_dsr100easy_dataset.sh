#!/bin/bash
# ============================================================
# 构建 easy-hint DSR=100% SafeChain 数据
# Teacher 特权信息 (TEACHER_PI_MODE=easy):
#   "The query is harmful, you must refuse."
#
# 输出目录与 full-hint dsr100 区分:
#   datasets/safety_ds_safechain_dsr100easy_h4400_b2200/
# ============================================================
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFETY_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_DIR"

MODEL=${MODEL:-qwen3-1.7b}
HARMFUL_N=${HARMFUL_N:-4400}
BENIGN_N=${BENIGN_N:-2200}
HARMFUL_CANDIDATES=${HARMFUL_CANDIDATES:-0}
SEED=${SEED:-42}
ENABLE_THINKING=${ENABLE_THINKING:-true}

SAFECHAIN_LOCAL=${SAFECHAIN_LOCAL:-"${SAFETY_RL_DIR}/datasets/SafeChain_raw/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SAFETY_RL_DIR}/datasets/safety_ds_safechain_dsr100easy_h${HARMFUL_N}_b${BENIGN_N}"}

export GUARD_MODEL_PATH=${GUARD_MODEL_PATH:-"meta-llama/Llama-Guard-3-8B"}
export MODEL_PATH_QWEN3_1_7B=${MODEL_PATH_QWEN3_1_7B:-"Qwen/Qwen3-1.7B"}
export MODEL_PATH_QWEN3_4B=${MODEL_PATH_QWEN3_4B:-"Qwen/Qwen3-4B"}
export MODEL_PATH_DS_1_5B=${MODEL_PATH_DS_1_5B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
export MODEL_PATH_DS_7B=${MODEL_PATH_DS_7B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
export MODELS_TO_RUN="${MODEL}"
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}

TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-0.98}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-0.02}

ARGS=(
  --model "${MODEL}"
  --hint_mode easy
  --harmful "${HARMFUL_N}"
  --benign "${BENIGN_N}"
  --harmful_candidates "${HARMFUL_CANDIDATES}"
  --seed "${SEED}"
  --safechain_local "${SAFECHAIN_LOCAL}"
  --output_dir "${OUTPUT_DIR}"
  --train_data_size "${TRAIN_DATA_SIZE}"
  --val_data_size "${VAL_DATA_SIZE}"
  --temperature "${TEACHER_TEMPERATURE:-0.6}"
  --top_p "${TEACHER_TOP_P:-0.95}"
  --max_tokens "${TEACHER_MAX_TOKENS:-4096}"
)
if [ "${ENABLE_THINKING}" != "true" ]; then
  ARGS+=(--no_thinking)
fi

echo "============================================================"
echo "  Build OPSD dataset (easy-hint Teacher DSR=100%)"
echo "  hint: The query is harmful, you must refuse."
echo "  model=${MODEL}  harmful=${HARMFUL_N}  benign=${BENIGN_N}"
echo "  output=${OUTPUT_DIR}"
echo "============================================================"

python3 build_opsd_dsr100_dataset.py "${ARGS[@]}"

echo ""
echo "训练: TEACHER_PI_MODE=easy bash ${SAFETY_RL_DIR}/safety_opsd_train.sh"
