#!/bin/bash
# ============================================================
# Teacher-on-Student-Prefix 续写实验
#
# 1) Student 在 DATASET_NAME(wildchat|wildjailbreak) 上 rollout（可复用）
# 2) LlamaGuard 评 Student；抽样 TARGET_UNSAFE_N 条 unsafe（默认 100）
# 3) 学生前缀拼到 Teacher 后续写；若 student_token_len<=L 则跳过并计失败
# 4) LlamaGuard 评 Teacher；主指标 Rescue=P(T safe|S unsafe ∧ T alone safe)
# ============================================================
set -euo pipefail
set -x

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "[INFO] Activate env first, e.g.: conda activate eopsa"
fi

# ============================================================
# 可调参数
# ============================================================

# GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 模型路径
export MODEL_PATH_QWEN3_1_7B=${MODEL_PATH_QWEN3_1_7B:-"Qwen/Qwen3-1.7B"}
export MODEL_PATH_QWEN3_4B=${MODEL_PATH_QWEN3_4B:-"Qwen/Qwen3-4B"}
export MODEL_PATH_DS_1_5B=${MODEL_PATH_DS_1_5B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
export MODEL_PATH_DS_7B=${MODEL_PATH_DS_7B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
export GUARD_MODEL_PATH=${GUARD_MODEL_PATH:-"meta-llama/Llama-Guard-3-8B"}

# 要评测的模型（逗号分割）
# 注意：本机若无 ds-1.5b 权重，可设 RUN_MODELS=qwen3-1.7b,qwen3-4b,ds-7b
RUN_MODELS=${RUN_MODELS:-"qwen3-1.7b,qwen3-4b,ds-1.5b,ds-7b"}
IFS=',' read -ra MODEL_LIST <<< "$RUN_MODELS"

# 数据集：直接用 eval_llm_safety（config.yaml + load_and_prep_data）
# 可选: wildchat | wildjailbreak | strongreject | harmbench | ...
export DATASET_NAME=${DATASET_NAME:-wildchat}
export DATASET_LIMIT=${DATASET_LIMIT:-${WILDCHAT_LIMIT:--1}}  # -1=全部
SCRIPT_DIR_EARLY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_EARLY="$(cd "${SCRIPT_DIR_EARLY}/../../.." && pwd)"
export EVAL_LLM_SAFETY_DIR=${EVAL_LLM_SAFETY_DIR:-"${REPO_ROOT_EARLY}/../LLM-Safety-Eval"}
export EVAL_LLM_SAFETY_CONFIG=${EVAL_LLM_SAFETY_CONFIG:-"${EVAL_LLM_SAFETY_DIR}/config.yaml"}

# 前缀长度；长度不够的 (sample,L) 计为教师失败，不再要求 len>最后一项
export PREFIX_LENGTHS=${PREFIX_LENGTHS:-"0,12,24,36,48,64,128,256,512,1024,2048,3072"}
export TARGET_UNSAFE_N=${TARGET_UNSAFE_N:-100}
export SAMPLE_SEED=${SAMPLE_SEED:-42}
export PREFIX_SOFT_MIN_LEN=${PREFIX_SOFT_MIN_LEN:-0}

# 强制重跑 Student（默认复用 outputs/.../{dataset}/*_student_rollouts.json）
export FORCE_STUDENT_ROLLOUT=${FORCE_STUDENT_ROLLOUT:-0}

# vLLM 引擎
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}

# Student 采样
export STUDENT_TEMPERATURE=${STUDENT_TEMPERATURE:-0.6}
export STUDENT_TOP_P=${STUDENT_TOP_P:-0.95}
export STUDENT_MAX_TOKENS=${STUDENT_MAX_TOKENS:-8192}

# Teacher 续写采样
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-0.6}
export TEACHER_TOP_P=${TEACHER_TOP_P:-0.95}
export TEACHER_MAX_TOKENS=${TEACHER_MAX_TOKENS:-8192}

# ============================================================

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUT_DIR="outputs/teacher_prefix_continue/${DATASET_NAME}"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  Teacher-on-Student-Prefix 续写实验"
echo "  模型:              ${MODEL_LIST[*]}"
echo "  GPU:               $CUDA_VISIBLE_DEVICES"
echo "  DATASET_NAME:      $DATASET_NAME"
echo "  DATASET_LIMIT:     $DATASET_LIMIT"
echo "  EVAL_LLM_SAFETY:   $EVAL_LLM_SAFETY_DIR"
echo "  PREFIX_LENGTHS:    $PREFIX_LENGTHS"
echo "  TARGET_UNSAFE_N:   $TARGET_UNSAFE_N"
echo "  SAMPLE_SEED:       $SAMPLE_SEED"
echo "  PREFIX_SOFT_MIN:   $PREFIX_SOFT_MIN_LEN"
echo "  FORCE_STUDENT:     $FORCE_STUDENT_ROLLOUT"
echo "  STUDENT_MAX_TOKENS:$STUDENT_MAX_TOKENS"
echo "  TEACHER_MAX_TOKENS:$TEACHER_MAX_TOKENS"
echo "  VLLM_MAX_MODEL_LEN:$VLLM_MAX_MODEL_LEN"
echo "  OUT_DIR:           $OUT_DIR"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for MODEL in "${MODEL_LIST[@]}"; do
    MODEL=$(echo "$MODEL" | xargs)
    if [ -z "$MODEL" ]; then
        continue
    fi
    echo ""
    echo "############################################################"
    echo "  >>> 开始: $MODEL @ $DATASET_NAME  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "############################################################"

    export MODELS_TO_RUN="$MODEL"
    python3 teacher_prefix_continue.py 2>&1 | tee "${OUT_DIR}/${MODEL}_run.log"

    echo "  <<< 完成: $MODEL  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "============================================"
echo "  运行 LlamaGuard 评测 (Teacher)"
echo "============================================"
python3 run_teacher_prefix_guard_eval.py 2>&1 | tee "${OUT_DIR}/guard_eval.log"

echo ""
echo "============================================"
echo "  最终 Rescue 汇总"
echo "============================================"
python3 -c "
import glob, json, os
out_dir = 'outputs/teacher_prefix_continue/${DATASET_NAME}'
files = sorted(glob.glob(os.path.join(out_dir, '*_prefix_pipeline.json')))
print(f'找到 {len(files)} 个结果文件 @ {out_dir}:')
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    name = os.path.basename(f).replace('_prefix_pipeline.json', '')
    summary = data.get('guard_summary')
    if not summary:
        print(f'  {name}: 尚无 Guard 结果')
        continue
    stu = summary['student']
    print(f'  {name}  n={stu[\"total\"]}  (student unsafe set)')
    for pkey, r in sorted(summary.get('trr_by_prefix', {}).items(), key=lambda x: int(x[0])):
        print(
            f'    Rescue@{pkey:>4s}  {r[\"rescue_rate\"]:.2%}  '
            f'(ok={r[\"success\"]}/{r[\"total\"]}; short_fail={r[\"n_short_as_fail\"]})'
        )
"

echo ""
echo "完成！结果目录: ${OUT_DIR}/"
echo "画图: python3 plot/plot_prefix_direction_compare.py --dataset ${DATASET_NAME}"
