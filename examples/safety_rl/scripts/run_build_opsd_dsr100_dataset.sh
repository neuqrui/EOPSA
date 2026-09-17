#!/bin/bash
# ============================================================
# 构建 Teacher DSR=100% 的 SafeChain 数据，供 OPSD 训练使用
#
# 1) 有害候选 → Teacher 推理 → Llama-Guard，只保留 safe
# 2) 取满 HARMFUL_N=2200；另抽 BENIGN_N=2200（1:1；良性不跑 DSR）
# 3) 写出 train.jsonl / val.jsonl（与 prepare_safechain 同 schema）
# ============================================================
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFETY_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_DIR"

MODEL=${MODEL:-qwen3-1.7b}
HARMFUL_N=${HARMFUL_N:-2200}
BENIGN_N=${BENIGN_N:-2200}
# 0 = 对全部有害池打分（~1.8万）；也可设更大候选以更容易凑满 4400
HARMFUL_CANDIDATES=${HARMFUL_CANDIDATES:-0}
SEED=${SEED:-42}
ENABLE_THINKING=${ENABLE_THINKING:-true}
USE_RESPONSE_HINT=${USE_RESPONSE_HINT:-false}

SAFECHAIN_LOCAL=${SAFECHAIN_LOCAL:-"${SAFETY_RL_DIR}/datasets/SafeChain_raw/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SAFETY_RL_DIR}/datasets/safety_ds_safechain_dsr100_h${HARMFUL_N}_b${BENIGN_N}"}

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
if [ "${USE_RESPONSE_HINT}" = "true" ]; then
  ARGS+=(--use_response_hint)
fi
if [ "${ENABLE_THINKING}" != "true" ]; then
  ARGS+=(--no_thinking)
fi

echo "============================================================"
echo "  Build OPSD dataset (Teacher DSR=100% harmful)"
echo "  model=${MODEL}  harmful=${HARMFUL_N}  benign=${BENIGN_N}"
echo "  candidates=${HARMFUL_CANDIDATES:-ALL}"
echo "  output=${OUTPUT_DIR}"
echo "============================================================"

python3 build_opsd_dsr100_dataset.py "${ARGS[@]}"

echo ""
echo "训练示例:"
echo "  cd ${SAFETY_RL_DIR}"
echo "  DATA_DIR=${OUTPUT_DIR} ENABLE_MATH_MIX=false REBUILD_DATA=0 bash safety_opsd_train.sh"
