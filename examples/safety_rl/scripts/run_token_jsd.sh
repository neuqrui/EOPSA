#!/bin/bash
# ============================================================
# Student–Teacher token-wise Top-K JSD 分析
# 与 safety_opsd_train.sh 中 topk_jsd (默认 k=512) 一致
# ============================================================
set -euo pipefail
set -x

# GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# 模型（默认四个）
export MODEL_PATH_QWEN3_1_7B=${MODEL_PATH_QWEN3_1_7B:-"Qwen/Qwen3-1.7B"}
export MODEL_PATH_QWEN3_4B=${MODEL_PATH_QWEN3_4B:-"Qwen/Qwen3-4B"}
export MODEL_PATH_DS_1_5B=${MODEL_PATH_DS_1_5B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
export MODEL_PATH_DS_7B=${MODEL_PATH_DS_7B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}

RUN_MODELS=${RUN_MODELS:-"qwen3-1.7b,qwen3-4b,ds-1.5b,ds-7b"}
IFS=',' read -ra MODEL_LIST <<< "$RUN_MODELS"

# 数据
export DATASET_NAME=${DATASET_NAME:-wildchat}
export DATASET_LIMIT=${DATASET_LIMIT:-${WILDCHAT_LIMIT:--1}}
# Clone https://github.com/neuqrui/LLM-Safety-Eval and point EVAL_LLM_SAFETY_DIR at it
# if you need WildChat / WildJailbreak loaders for analysis scripts.
export EVAL_LLM_SAFETY_DIR=${EVAL_LLM_SAFETY_DIR:-}

# Top-K JSD（对齐 training）
export DISTILLATION_TOPK=${DISTILLATION_TOPK:-512}
export JSD_MAX_TOKENS=${JSD_MAX_TOKENS:-1024}   # 分析截断；-1=全长
export FORCE_STUDENT_ROLLOUT=${FORCE_STUDENT_ROLLOUT:-0}
export FORCE_JSD=${FORCE_JSD:-0}

# vLLM / Student rollout
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.5}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
export STUDENT_TEMPERATURE=${STUDENT_TEMPERATURE:-0.6}
export STUDENT_TOP_P=${STUDENT_TOP_P:-0.95}
export STUDENT_MAX_TOKENS=${STUDENT_MAX_TOKENS:-8192}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "outputs/token_jsd/${DATASET_NAME}"

echo "============================================"
echo "  Token-wise Top-K JSD"
echo "  MODELS:            ${MODEL_LIST[*]}"
echo "  DATASET:           $DATASET_NAME  limit=$DATASET_LIMIT"
echo "  DISTILLATION_TOPK: $DISTILLATION_TOPK"
echo "  JSD_MAX_TOKENS:    $JSD_MAX_TOKENS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for MODEL in "${MODEL_LIST[@]}"; do
    MODEL=$(echo "$MODEL" | xargs)
    [ -z "$MODEL" ] && continue
    echo ""
    echo "############################################################"
    echo "  >>> $MODEL  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "############################################################"
    export MODELS_TO_RUN="$MODEL"
    python3 token_jsd_analyze.py 2>&1 | tee "outputs/token_jsd/${DATASET_NAME}/${MODEL}_run.log"
done

echo ""
echo "============================================"
echo "  绘图"
echo "============================================"
python3 plot/plot_token_jsd.py --dataset "$DATASET_NAME" \
    --models "qwen3-1.7b,qwen3-4b,ds-1.5b,ds-7b"

echo ""
echo "完成: outputs/token_jsd/${DATASET_NAME}/"
echo "图:   plot/figures/token_jsd_${DATASET_NAME}.*"
