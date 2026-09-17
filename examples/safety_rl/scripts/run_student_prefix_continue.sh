#!/bin/bash
# ============================================================
# Student-on-Teacher-Prefix 续写实验（与 teacher_prefix_continue 反向）
#
# 1) Teacher 带特权模版 rollout（可复用）
# 2) LlamaGuard 评 Teacher；并与 Student 独立失败样本取交集，抽样 TARGET_SAFE_N（默认 100）
#    评测集 = teacher safe ∩ student alone unsafe
# 3) 教师前缀拼到 Student 后续写；若 teacher_token_len<=L 则计防守成功（短回复已结束）
# 4) LlamaGuard 评 Student；主指标 Hold=P(S safe | T safe ∧ S alone fail)
# ============================================================
set -euo pipefail
set -x

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "[INFO] Activate env first, e.g.: conda activate eopsa"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export MODEL_PATH_QWEN3_1_7B=${MODEL_PATH_QWEN3_1_7B:-"Qwen/Qwen3-1.7B"}
export MODEL_PATH_QWEN3_4B=${MODEL_PATH_QWEN3_4B:-"Qwen/Qwen3-4B"}
export MODEL_PATH_DS_1_5B=${MODEL_PATH_DS_1_5B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
export MODEL_PATH_DS_7B=${MODEL_PATH_DS_7B:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
export GUARD_MODEL_PATH=${GUARD_MODEL_PATH:-"meta-llama/Llama-Guard-3-8B"}

# 本机若无 ds-1.5b 权重，可设 RUN_MODELS=qwen3-1.7b,qwen3-4b,ds-7b
RUN_MODELS=${RUN_MODELS:-"qwen3-1.7b,qwen3-4b,ds-1.5b,ds-7b"}
IFS=',' read -ra MODEL_LIST <<< "$RUN_MODELS"

export DATASET_NAME=${DATASET_NAME:-wildchat}
export DATASET_LIMIT=${DATASET_LIMIT:-${WILDCHAT_LIMIT:--1}}
SCRIPT_DIR_EARLY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_EARLY="$(cd "${SCRIPT_DIR_EARLY}/../../.." && pwd)"
export EVAL_LLM_SAFETY_DIR=${EVAL_LLM_SAFETY_DIR:-"${REPO_ROOT_EARLY}/../LLM-Safety-Eval"}
export EVAL_LLM_SAFETY_CONFIG=${EVAL_LLM_SAFETY_CONFIG:-"${EVAL_LLM_SAFETY_DIR}/config.yaml"}

export PREFIX_LENGTHS=${PREFIX_LENGTHS:-"0,12,24,36,48,64,128,256,512,1024,2048,3072"}
export TARGET_SAFE_N=${TARGET_SAFE_N:-${TARGET_UNSAFE_N:-100}}
export SAMPLE_SEED=${SAMPLE_SEED:-42}
export PREFIX_SOFT_MIN_LEN=${PREFIX_SOFT_MIN_LEN:-0}
export FORCE_TEACHER_ROLLOUT=${FORCE_TEACHER_ROLLOUT:-0}

export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}

export STUDENT_TEMPERATURE=${STUDENT_TEMPERATURE:-0.6}
export STUDENT_TOP_P=${STUDENT_TOP_P:-0.95}
export STUDENT_MAX_TOKENS=${STUDENT_MAX_TOKENS:-8192}

export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-0.6}
export TEACHER_TOP_P=${TEACHER_TOP_P:-0.95}
export TEACHER_MAX_TOKENS=${TEACHER_MAX_TOKENS:-8192}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUT_DIR="outputs/student_prefix_continue/${DATASET_NAME}"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  Student-on-Teacher-Prefix 续写（反向）"
echo "  模型:              ${MODEL_LIST[*]}"
echo "  GPU:               $CUDA_VISIBLE_DEVICES"
echo "  DATASET_NAME:      $DATASET_NAME"
echo "  DATASET_LIMIT:     $DATASET_LIMIT"
echo "  EVAL_LLM_SAFETY:   $EVAL_LLM_SAFETY_DIR"
echo "  PREFIX_LENGTHS:    $PREFIX_LENGTHS"
echo "  TARGET_SAFE_N:     $TARGET_SAFE_N"
echo "  SAMPLE_SEED:       $SAMPLE_SEED"
echo "  PREFIX_SOFT_MIN:   $PREFIX_SOFT_MIN_LEN"
echo "  FORCE_TEACHER:     $FORCE_TEACHER_ROLLOUT"
echo "  OUT_DIR:           $OUT_DIR"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for MODEL in "${MODEL_LIST[@]}"; do
    MODEL=$(echo "$MODEL" | xargs)
    [ -z "$MODEL" ] && continue
    echo ""
    echo "############################################################"
    echo "  >>> 开始: $MODEL @ $DATASET_NAME  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "############################################################"
    export MODELS_TO_RUN="$MODEL"
    python3 student_prefix_continue.py 2>&1 | tee "${OUT_DIR}/${MODEL}_run.log"
    echo "  <<< 完成: $MODEL  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "============================================"
echo "  运行 LlamaGuard 评测 (Student)"
echo "============================================"
python3 run_student_prefix_guard_eval.py 2>&1 | tee "${OUT_DIR}/guard_eval.log"

echo ""
echo "============================================"
echo "  最终 Hold 汇总"
echo "============================================"
python3 -c "
import glob, json, os
out_dir = 'outputs/student_prefix_continue/${DATASET_NAME}'
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
    tea = summary['teacher']
    print(f'  {name}  n={tea[\"total\"]}  (teacher safe set, DSR={tea[\"DSR\"]:.2%})')
    for pkey, r in sorted(summary.get('hold_by_prefix', {}).items(), key=lambda x: int(x[0])):
        print(
            f'    Hold@{pkey:>4s}  {r[\"hold_rate\"]:.2%}  '
            f'(ok={r[\"success\"]}/{r[\"total\"]}; short_ok={r[\"n_short_as_success\"]})'
        )
"

echo ""
echo "完成！结果目录: ${OUT_DIR}/"
echo "画图: python3 plot/plot_prefix_direction_compare.py --dataset ${DATASET_NAME}"
