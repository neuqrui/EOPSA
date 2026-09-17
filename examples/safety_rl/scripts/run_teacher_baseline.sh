#!/bin/bash
# ============================================================
# Teacher Baseline 评测
# 在 WildChat 上使用 safety_teacher.jinja（静态 HARMFUL_GUIDANCE）对模型逐个推理，
# 然后用 Llama-Guard-3-8B 评测 DSR/ASR。
# 逐模型循环执行，单个模型卡住不影响后续。
# ============================================================
set -euo pipefail
set -x

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "[INFO] Activate env first, e.g.: conda activate eopsa"
fi

# ============================================================
# 可调参数（修改这里的默认值即可，或运行时通过环境变量覆盖）
# ============================================================

# GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# 模型路径
export MODEL_PATH_QWEN3_1_7B=${MODEL_PATH_QWEN3_1_7B:-"Qwen/Qwen3-1.7B"}
export MODEL_PATH_QWEN3_4B=${MODEL_PATH_QWEN3_4B:-"Qwen/Qwen3-4B"}
export MODEL_PATH_DS_1_5B=${MODEL_PATH_DS_1_5B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
export MODEL_PATH_DS_7B=${MODEL_PATH_DS_7B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
export GUARD_MODEL_PATH=${GUARD_MODEL_PATH:-"meta-llama/Llama-Guard-3-8B"}

# 要评测的模型（逗号分割），默认全部。shell 将逐个循环执行。
RUN_MODELS=${RUN_MODELS:-"qwen3-1.7b,qwen3-4b,ds-1.5b,ds-7b"}
IFS=',' read -ra MODEL_LIST <<< "$RUN_MODELS"

# 数据集
export WILDCHAT_LIMIT=${WILDCHAT_LIMIT:-100}       # -1=全部数据

# vLLM 引擎
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}

# Teacher 采样参数
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-0.6}
export TEACHER_TOP_P=${TEACHER_TOP_P:-0.95}
export TEACHER_MAX_TOKENS=${TEACHER_MAX_TOKENS:-8192}

# ============================================================

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Teacher Baseline 评测"
echo "  逐个模型循环执行: ${MODEL_LIST[*]}"
echo "  GPU:               $CUDA_VISIBLE_DEVICES"
echo "  Qwen3-1.7B:        $MODEL_PATH_QWEN3_1_7B"
echo "  Qwen3-4B:          $MODEL_PATH_QWEN3_4B"
echo "  DS-1.5B:           $MODEL_PATH_DS_1_5B"
echo "  DS-7B:             $MODEL_PATH_DS_7B"
echo "  Guard Model:       $GUARD_MODEL_PATH"
echo "  WILDCHAT_LIMIT:    $WILDCHAT_LIMIT"
echo "  TEACHER_TEMP:      $TEACHER_TEMPERATURE"
echo "  TEACHER_TOP_P:     $TEACHER_TOP_P"
echo "  TEACHER_MAX_TOKENS:$TEACHER_MAX_TOKENS"
echo "  VLLM_GPU_MEM:      $VLLM_GPU_MEMORY_UTILIZATION"
echo "  VLLM_MAX_MODEL_LEN:$VLLM_MAX_MODEL_LEN"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ============================================================
# 逐模型循环执行
# ============================================================
for MODEL in "${MODEL_LIST[@]}"; do
    MODEL=$(echo "$MODEL" | xargs)  # trim whitespace
    if [ -z "$MODEL" ]; then
        continue
    fi
    echo ""
    echo "############################################################"
    echo "  >>> 开始评测模型: $MODEL"
    echo "  >>> 时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "############################################################"

    export MODELS_TO_RUN="$MODEL"
    python3 teacher_baseline.py 2>&1 | tee "outputs/teacher_baseline/${MODEL}_run.log"

    echo ""
    echo "  <<< 模型 $MODEL 评测完成 ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "============================================"
echo "  运行 Guard 评测 (Llama-Guard-3-8B):"
echo "============================================"
python3 run_teacher_guard_eval.py 2>&1 | tee "outputs/teacher_baseline/guard_eval.log"

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================"
echo "  最终安全评测汇总:"
echo "============================================"
python3 -c "
import glob, json, os
out_dir = 'outputs/teacher_baseline'
files = sorted(glob.glob(os.path.join(out_dir, '*_guard_results.json')))
print(f'找到 {len(files)} 个结果文件:')
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    total = len(data)
    unsafe = 0
    for d in data:
        for label in d.get('guard_labels', []):
            if label:
                unsafe += 1
    name = os.path.basename(f).replace('_guard_results.json', '')
    asr = unsafe / total if total > 0 else 0
    print(f'  {name:20s}  samples={total:3d}  ASR={asr:.2%}  DSR={1-asr:.2%}')
"

echo ""
echo "完成！结果: outputs/teacher_baseline/"
