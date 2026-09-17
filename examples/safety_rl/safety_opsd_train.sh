#!/bin/bash
# EOPSA training launcher (Adaptive Rollout Scheduling + Selective Distillation).
# Paper defaults: ARS on, Selective Distillation on, forward KL, 200 steps.
# Validation (VAL_METHOD):
#   llamaguard -> standalone vLLM Llama-Guard on a free GPU + HTTP judge  [default]
#   reward     -> safety_reward.py (API/rule judge; requires OPENAI_API_KEY)
#
# Single-machine launcher: opsd_main auto-starts Ray via ray.init() — no manual ray start.
set -euo pipefail
set -x

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "[INFO] Activate env first, e.g.: conda activate eopsa"
fi

export TOKENIZERS_PARALLELISM=false
export RAY_worker_num_grpc_internal_threads=1
export RAY_ADDRESS=""   # force a fresh local Ray cluster (opsd_main calls ray.init)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800000
export NCCL_DEBUG=WARN
export VERL_LOG_LEVEL=INFO

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
# vLLM TP; must divide NPROC_PER_NODE. 32B: 4gpu→4, 8gpu→2 or 4.
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG_PATH=${CONFIG_PATH:-"${SCRIPT_DIR}/safety_opsd_config.yaml"}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HUGGINGFACE_HUB_ENDPOINT=${HUGGINGFACE_HUB_ENDPOINT:-$HF_ENDPOINT}
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
DATA_SOURCE=${DATA_SOURCE:-safechain}
# Math mix / dynamic-PI builders are not shipped in the official release.
if [ "${ENABLE_MATH_MIX:-false}" = "true" ]; then
    echo "[ERROR] ENABLE_MATH_MIX is not supported in the official EOPSA release." >&2
    exit 1
fi
if [ "${USE_OPSD_DYNAMIC_PI:-false}" = "true" ]; then
    echo "[ERROR] USE_OPSD_DYNAMIC_PI is not supported in the official EOPSA release." >&2
    exit 1
fi
case "${DATA_SOURCE}" in
    safechain|legacy|star1) ;;
    *)
        echo "[ERROR] DATA_SOURCE must be safechain|legacy|star1, got: ${DATA_SOURCE}" >&2
        exit 1
        ;;
esac

HARMFUL_JSON=${HARMFUL_JSON:-"${SCRIPT_DIR}/datasets/harmful_risk_7B.json"}
BENIGN_JSON=${BENIGN_JSON:-"${SCRIPT_DIR}/datasets/benign_instructions.json"}
if [ "${DATA_SOURCE}" = "star1" ]; then
    # STAR-1 pool: up to 1000 harmful / 915 benign; default 2:1 like safechain (4400:2200).
    HARMFUL_N=${HARMFUL_N:-1000}
    BENIGN_N=${BENIGN_N:-500}
else
    HARMFUL_N=${HARMFUL_N:-4400}
    BENIGN_N=${BENIGN_N:-2200}
fi
# Math mix (OPSD Openthoughts): set ENABLE_MATH_MIX=true and MATH_N>0 (or MIX_RATIOS+MIX_TOTAL).
ENABLE_MATH_MIX=${ENABLE_MATH_MIX:-false}
MATH_N=${MATH_N:-0}
MATH_DATASET=${MATH_DATASET:-siyanzhao/Openthoughts_math_30k_opsd}
MATH_LOCAL_PATH=${MATH_LOCAL_PATH:-"${SCRIPT_DIR}/datasets/Openthoughts_math_30k_opsd/data"}
# Optional ratio mode, e.g. MIX_RATIOS="math:1,safety:2,benign:1" MIX_TOTAL=8800
MIX_RATIOS=${MIX_RATIOS:-}
MIX_TOTAL=${MIX_TOTAL:-0}
MATH_IN_VAL=${MATH_IN_VAL:-false}
# For DSR100 rebuilds use scripts/run_build_opsd_dsr100_dataset.sh (defaults 0.98/0.02 → val≥100).
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-0.98}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-0.02}
SEED=${SEED:-42}

SAFECHAIN_DATASET=${SAFECHAIN_DATASET:-UWNSL/SafeChain}
SAFECHAIN_LOCAL_PATH=${SAFECHAIN_LOCAL_PATH:-"${SCRIPT_DIR}/datasets/SafeChain_raw/data"}

STAR1_HARMFUL_DATASET=${STAR1_HARMFUL_DATASET:-UCSC-VLAA/STAR-1}
STAR1_BENIGN_DATASET=${STAR1_BENIGN_DATASET:-UCSC-VLAA/STAR-benign-915}
STAR1_LOCAL_PATH=${STAR1_LOCAL_PATH:-"${SCRIPT_DIR}/datasets/STAR1_raw/STAR-1.json"}
STAR1_BENIGN_LOCAL_PATH=${STAR1_BENIGN_LOCAL_PATH:-"${SCRIPT_DIR}/datasets/STAR1_raw/STAR-benign-915.json"}

# Default: Teacher+LlamaGuard DSR=100% filtered SafeChain.
# Teacher privileged information (PI) for harmful samples:
#   TEACHER_PI_MODE=hard — long guidance (dsr100 dataset)
#   TEACHER_PI_MODE=easy — "The query is harmful, you must refuse." (dsr100easy)
# If TEACHER_PI_MODE is unset, legacy flags still work:
#   USE_DSR100_DATA=true / USE_DSR100EASY_DATA=true
# Override: DATA_DIR=/other/path bash safety_opsd_train.sh
if [ "${DATA_SOURCE}" = "safechain" ]; then
    USE_DSR100_DATA=${USE_DSR100_DATA:-true}
    USE_DSR100EASY_DATA=${USE_DSR100EASY_DATA:-false}
else
    USE_DSR100_DATA=${USE_DSR100_DATA:-false}
    USE_DSR100EASY_DATA=${USE_DSR100EASY_DATA:-false}
fi
if [ -n "${TEACHER_PI_MODE:-}" ]; then
    case "${TEACHER_PI_MODE}" in
        hard|full)
            TEACHER_PI_MODE=hard
            USE_DSR100EASY_DATA=false
            if [ "${DATA_SOURCE}" = "safechain" ]; then
                USE_DSR100_DATA=true
            else
                USE_DSR100_DATA=false
            fi
            ;;
        easy)
            TEACHER_PI_MODE=easy
            if [ "${DATA_SOURCE}" = "safechain" ]; then
                USE_DSR100EASY_DATA=true
                USE_DSR100_DATA=false
            else
                USE_DSR100EASY_DATA=false
                USE_DSR100_DATA=false
            fi
            ;;
        *)
            echo "[ERROR] TEACHER_PI_MODE must be hard|easy, got: ${TEACHER_PI_MODE}" >&2
            exit 1
            ;;
    esac
elif [ "${USE_DSR100EASY_DATA}" = "true" ]; then
    TEACHER_PI_MODE=easy
else
    TEACHER_PI_MODE=hard
fi
USE_OPSD_DYNAMIC_PI=${USE_OPSD_DYNAMIC_PI:-false}
OPSD_DYNAMIC_PI_FACTORS_FILE=${OPSD_DYNAMIC_PI_FACTORS_FILE:-"${SCRIPT_DIR}/scripts/outputs/harm_factors/harmful_factors_h4400_gpt4o_mini.jsonl"}
if [ "${USE_DSR100EASY_DATA}" = "true" ]; then
    OPSD_DYNAMIC_PI_BASE_DIR_DEFAULT="${SCRIPT_DIR}/datasets/safety_ds_safechain_dsr100easy_h${HARMFUL_N}_b${BENIGN_N}"
elif [ "${USE_DSR100_DATA}" = "true" ]; then
    OPSD_DYNAMIC_PI_BASE_DIR_DEFAULT="${SCRIPT_DIR}/datasets/safety_ds_safechain_dsr100_h${HARMFUL_N}_b${BENIGN_N}"
elif [ "${DATA_SOURCE}" = "safechain" ]; then
    OPSD_DYNAMIC_PI_BASE_DIR_DEFAULT="${SCRIPT_DIR}/datasets/safety_ds_safechain_h${HARMFUL_N}_b${BENIGN_N}"
elif [ "${DATA_SOURCE}" = "star1" ]; then
    OPSD_DYNAMIC_PI_BASE_DIR_DEFAULT="${SCRIPT_DIR}/datasets/safety_ds_star1_h${HARMFUL_N}_b${BENIGN_N}"
else
    OPSD_DYNAMIC_PI_BASE_DIR_DEFAULT="${SCRIPT_DIR}/datasets/safety_ds_7B_hb_h${HARMFUL_N}_b${BENIGN_N}"
fi
OPSD_DYNAMIC_PI_BASE_DIR=${OPSD_DYNAMIC_PI_BASE_DIR:-"${OPSD_DYNAMIC_PI_BASE_DIR_DEFAULT}"}
if [ -z "${DATA_DIR:-}" ]; then
    if [ "${USE_OPSD_DYNAMIC_PI}" = "true" ]; then
        DATA_DIR="${OPSD_DYNAMIC_PI_BASE_DIR}_opsddynpi"
    elif [ "${USE_DSR100EASY_DATA}" = "true" ]; then
        DATA_DIR="${SCRIPT_DIR}/datasets/safety_ds_safechain_dsr100easy_h${HARMFUL_N}_b${BENIGN_N}"
    elif [ "${USE_DSR100_DATA}" = "true" ]; then
        DATA_DIR="${SCRIPT_DIR}/datasets/safety_ds_safechain_dsr100_h${HARMFUL_N}_b${BENIGN_N}"
    elif [ "${ENABLE_MATH_MIX}" = "true" ]; then
        if [ -n "${MIX_RATIOS}" ] && [ "${MIX_TOTAL}" -gt 0 ] 2>/dev/null; then
            RATIO_TAG=$(echo "${MIX_RATIOS}" | tr ':,' '__' | tr -cd '[:alnum:]_')
            DATA_DIR="${SCRIPT_DIR}/datasets/opsd_mix_${DATA_SOURCE}_${RATIO_TAG}_n${MIX_TOTAL}"
        else
            DATA_DIR="${SCRIPT_DIR}/datasets/opsd_mix_${DATA_SOURCE}_h${HARMFUL_N}_b${BENIGN_N}_m${MATH_N}"
        fi
    elif [ "${DATA_SOURCE}" = "safechain" ]; then
        DATA_DIR="${SCRIPT_DIR}/datasets/safety_ds_safechain_h${HARMFUL_N}_b${BENIGN_N}"
    elif [ "${DATA_SOURCE}" = "star1" ]; then
        DATA_DIR="${SCRIPT_DIR}/datasets/safety_ds_star1_h${HARMFUL_N}_b${BENIGN_N}"
    else
        DATA_DIR="${SCRIPT_DIR}/datasets/safety_ds_7B_hb_h${HARMFUL_N}_b${BENIGN_N}"
    fi
fi
TRAIN_DATA=${TRAIN_DATA:-"${DATA_DIR}/train.jsonl"}
VAL_DATA=${VAL_DATA:-"${DATA_DIR}/val.jsonl"}

# Optional: replace val harmful with WildJailbreak + WildChat mix (keeps current benign).
# Val mix is shared across TEACHER_PI_MODE (easy/hard only changes train safe_reference).
# Default: reuse the existing hard-dsr100 mix; do not rebuild per PI mode.
# Mutually exclusive with USE_WILDCHAT100_VAL_HARMFUL.
if [ "${DATA_SOURCE}" = "safechain" ] && [ "${USE_DSR100_DATA}" = "true" ]; then
    USE_WJWC_VAL_HARMFUL=${USE_WJWC_VAL_HARMFUL:-true}
    WJWC_VAL_MIX_DEFAULT="${SCRIPT_DIR}/datasets/safety_ds_safechain_dsr100_h${HARMFUL_N}_b${BENIGN_N}/val_wjwc_mix.jsonl"
else
    USE_WJWC_VAL_HARMFUL=${USE_WJWC_VAL_HARMFUL:-false}
    WJWC_VAL_MIX_DEFAULT="${DATA_DIR}/val_wjwc_mix.jsonl"
fi
USE_WJWC_VAL_HARMFUL=${USE_WJWC_VAL_HARMFUL}
WJWC_VAL_MIX=${WJWC_VAL_MIX:-"${WJWC_VAL_MIX_DEFAULT}"}
WJWC_RATIO=${WJWC_RATIO:-0.5}
WJWC_REBUILD=${WJWC_REBUILD:-false}

# Optional: replace val harmful with the exact WildChat-100 set used in prefix/plotting
# (scripts/outputs/student_prefix_continue/wildchat/qwen3-4b_prefix_pipeline.json).
# Mix keeps current val benign (overreject) rows. Prepared under datasets/ars_wildchat100/.
USE_WILDCHAT100_VAL_HARMFUL=${USE_WILDCHAT100_VAL_HARMFUL:-false}
WILDCHAT100_VAL_MIX=${WILDCHAT100_VAL_MIX:-"${SCRIPT_DIR}/datasets/ars_wildchat100/val_with_benign.jsonl"}
WILDCHAT100_VAL_HARM=${WILDCHAT100_VAL_HARM:-"${SCRIPT_DIR}/datasets/ars_wildchat100/val_harmful.jsonl"}
if [ "${USE_WJWC_VAL_HARMFUL}" = "true" ] && [ "${USE_WILDCHAT100_VAL_HARMFUL}" = "true" ]; then
    echo "[ERROR] Set only one of USE_WJWC_VAL_HARMFUL / USE_WILDCHAT100_VAL_HARMFUL" >&2
    exit 1
fi
if [ "${USE_WJWC_VAL_HARMFUL}" = "true" ]; then
    if [ "${WJWC_REBUILD}" = "true" ] || [ ! -f "${WJWC_VAL_MIX}" ]; then
        _WJWC_BASE_VAL="${VAL_DATA}"
        _WJWC_BASE_TRAIN="${TRAIN_DATA}"
        if [ ! -f "${_WJWC_BASE_VAL}" ] && [ "${USE_DSR100_DATA}" = "true" ]; then
            _WJWC_BASE_VAL="${SCRIPT_DIR}/datasets/safety_ds_safechain_dsr100_h${HARMFUL_N}_b${BENIGN_N}/val.jsonl"
            _WJWC_BASE_TRAIN="${SCRIPT_DIR}/datasets/safety_ds_safechain_dsr100_h${HARMFUL_N}_b${BENIGN_N}/train.jsonl"
        fi
        if [ ! -f "${_WJWC_BASE_VAL}" ]; then
            echo "[ERROR] base VAL missing for WJWC mix: ${_WJWC_BASE_VAL}" >&2
            exit 1
        fi
        echo "[INFO] Building WJ+WC val mix -> ${WJWC_VAL_MIX} (wj_ratio=${WJWC_RATIO})"
        python3 "${SCRIPT_DIR}/scripts/build_val_wjwc_mix.py" \
            --val "${_WJWC_BASE_VAL}" \
            --train "${_WJWC_BASE_TRAIN}" \
            --wj_ratio "${WJWC_RATIO}" \
            --output "${WJWC_VAL_MIX}" \
            --no_backup
    fi
    if [ ! -f "${WJWC_VAL_MIX}" ]; then
        echo "[ERROR] WJWC val mix missing: ${WJWC_VAL_MIX}" >&2
        exit 1
    fi
    VAL_DATA="${WJWC_VAL_MIX}"
    echo "[INFO] VAL harmful = WildJailbreak+WildChat mix (+ current benign) -> ${VAL_DATA}"
fi
if [ "${USE_WILDCHAT100_VAL_HARMFUL}" = "true" ]; then
    if [ ! -f "${WILDCHAT100_VAL_MIX}" ]; then
        echo "[ERROR] WildChat100 val mix missing: ${WILDCHAT100_VAL_MIX}" >&2
        echo "  Rebuild with the prepare snippet in datasets/ars_wildchat100/" >&2
        exit 1
    fi
    VAL_DATA="${WILDCHAT100_VAL_MIX}"
    echo "[INFO] VAL harmful = WildChat plot-100 (+ current benign) -> ${VAL_DATA}"
fi

# Templates: mixed routes by problem_type (math from opsd/data_collator.py).
if [ "${ENABLE_MATH_MIX}" = "true" ]; then
    FORMAT_PROMPT=${FORMAT_PROMPT:-"${SCRIPT_DIR}/format_prompt/mixed_student.jinja"}
    TEACHER_TEMPLATE=${TEACHER_TEMPLATE:-"${SCRIPT_DIR}/format_prompt/mixed_teacher.jinja"}
else
    FORMAT_PROMPT=${FORMAT_PROMPT:-"${SCRIPT_DIR}/format_prompt/safety_student.jinja"}
    TEACHER_TEMPLATE=${TEACHER_TEMPLATE:-"${SCRIPT_DIR}/format_prompt/safety_teacher.jinja"}
fi
TRAIN_REWARD_FUNCTION="${SCRIPT_DIR}/reward/opsd_noop_reward.py:compute_score"

# Val judge: reward | llamaguard
VAL_METHOD=${VAL_METHOD:-llamaguard}
# LlamaGuard: standalone vLLM on a dedicated GPU (HTTP). Must NOT overlap training GPUs.
GUARD_MODEL_PATH=${GUARD_MODEL_PATH:-"meta-llama/Llama-Guard-3-8B"}
GUARD_BACKEND=${GUARD_BACKEND:-http}          # http (recommended) | hf (debug only)
GUARD_PORT=${GUARD_PORT:-8123}
GUARD_HOST=${GUARD_HOST:-127.0.0.1}
GUARD_BASE_URL=${GUARD_BASE_URL:-"http://${GUARD_HOST}:${GUARD_PORT}/v1"}
GUARD_SERVED_NAME=${GUARD_SERVED_NAME:-"$(basename "${GUARD_MODEL_PATH}")"}
GUARD_AUTO_START=${GUARD_AUTO_START:-true}    # start vLLM if not already up
GUARD_STOP_AFTER=${GUARD_STOP_AFTER:-false}   # kill the server we started when train exits
# Guard vLLM memory fraction of the *whole* GPU (not free mem). Default 0.90.
# Lower (e.g. 0.25~0.35) only helps if you intentionally co-locate; also lower
# worker.rollout.gpu_memory_utilization and allow GUARD_GPU overlap (see below).
GUARD_GPU_MEM_UTIL=${GUARD_GPU_MEM_UTIL:-0.8}
GUARD_MAX_MODEL_LEN=${GUARD_MAX_MODEL_LEN:-32768}
# Set GUARD_ALLOW_TRAIN_GPU_OVERLAP=true to skip the GUARD_GPU vs train GPU check
# (risky: Llama-Guard-3-8B weights alone need ~16GB; co-locate only on large GPUs).
GUARD_ALLOW_TRAIN_GPU_OVERLAP=${GUARD_ALLOW_TRAIN_GPU_OVERLAP:-true}

GUARD_GPU_AUTO=${GUARD_GPU_AUTO:-false}
if [ "${GUARD_GPU_AUTO}" = "true" ] && [ -z "${GUARD_GPU:-}" ]; then
    GUARD_GPU=""
    IFS=',' read -ra _TRAIN_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
    for _g in 5 4 3 2 1 0 6 7; do
        _taken=false
        for _t in "${_TRAIN_GPUS[@]:-}"; do
            if [ "${_t}" = "${_g}" ]; then _taken=true; break; fi
        done
        if [ "${_taken}" = false ]; then GUARD_GPU="${_g}"; break; fi
    done
    GUARD_GPU=${GUARD_GPU:-6}
fi
GUARD_GPU=${GUARD_GPU:-7}
# format_mode for val: reward path keeps FORMAT_MODE; llamaguard defaults to off (pure DSR).
VAL_FORMAT_MODE=${VAL_FORMAT_MODE:-}

ENABLE_THINKING=${ENABLE_THINKING:-true}
FORMAT_MODE=${FORMAT_MODE:-api_overall}
GUARD_STARTED_BY_US=0
GUARD_PID_FILE=""
if [ "${VAL_METHOD}" = "llamaguard" ]; then
    VAL_REWARD_FUNCTION="${SCRIPT_DIR}/reward/llamaguard_reward.py:compute_score"
    VAL_FORMAT_MODE=${VAL_FORMAT_MODE:-off}
    export GUARD_MODEL_PATH GUARD_BASE_URL GUARD_BACKEND
    # Sanity: Guard GPU must not be in the training visible set (unless explicitly allowed).
    # Skip entirely when VAL_FREQ=0 (no validation → no Guard needed).
    if [ "${VAL_FREQ:-50}" != "0" ] && [ "${GUARD_ALLOW_TRAIN_GPU_OVERLAP}" != "true" ]; then
        IFS=',' read -ra _TRAIN_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
        for _t in "${_TRAIN_GPUS[@]:-}"; do
            if [ "${_t}" = "${GUARD_GPU}" ]; then
                echo "[ERROR] GUARD_GPU=${GUARD_GPU} overlaps CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
                echo "  Pick another free GPU, e.g. GUARD_GPU=5" >&2
                echo "  Or set GUARD_ALLOW_TRAIN_GPU_OVERLAP=true and lower GUARD_GPU_MEM_UTIL + rollout util." >&2
                exit 1
            fi
        done
    elif [ "${VAL_FREQ:-50}" != "0" ] && [ "${GUARD_ALLOW_TRAIN_GPU_OVERLAP}" = "true" ]; then
        echo "[WARN] GUARD_ALLOW_TRAIN_GPU_OVERLAP=true: Guard may share a training GPU (mem_util=${GUARD_GPU_MEM_UTIL})"
    fi
    if [ "${VAL_FREQ:-50}" != "0" ] && [ "${GUARD_BACKEND}" = "http" ] && [ "${GUARD_AUTO_START}" = "true" ]; then
        echo "[INFO] Ensuring LlamaGuard vLLM on GPU ${GUARD_GPU} port ${GUARD_PORT} (mem_util=${GUARD_GPU_MEM_UTIL}) ..."
        GUARD_GPU="${GUARD_GPU}" GUARD_PORT="${GUARD_PORT}" GUARD_HOST="${GUARD_HOST}" \
            GUARD_MODEL_PATH="${GUARD_MODEL_PATH}" GUARD_SERVED_NAME="${GUARD_SERVED_NAME}" \
            GUARD_GPU_MEM_UTIL="${GUARD_GPU_MEM_UTIL}" GUARD_MAX_MODEL_LEN="${GUARD_MAX_MODEL_LEN}" \
            bash "${SCRIPT_DIR}/scripts/start_llamaguard_vllm.sh"
        GUARD_PID_FILE="${SCRIPT_DIR}/logs/llamaguard_vllm/guard_gpu${GUARD_GPU}_p${GUARD_PORT}.pid"
        if [ -f "${GUARD_PID_FILE}.status" ] && [ "$(cat "${GUARD_PID_FILE}.status")" = "STARTED" ]; then
            GUARD_STARTED_BY_US=1
        fi
    elif [ "${VAL_FREQ:-50}" = "0" ]; then
        echo "[INFO] VAL_FREQ=0 → skip LlamaGuard (validation disabled)"
    fi
    cleanup_guard() {
        if [ "${GUARD_STOP_AFTER}" = "true" ] && [ "${GUARD_STARTED_BY_US}" = "1" ] && [ -n "${GUARD_PID_FILE}" ] && [ -f "${GUARD_PID_FILE}" ]; then
            _pid=$(cat "${GUARD_PID_FILE}" || true)
            if [ -n "${_pid}" ] && kill -0 "${_pid}" 2>/dev/null; then
                echo "[INFO] Stopping LlamaGuard vLLM pid=${_pid}"
                kill "${_pid}" 2>/dev/null || true
            fi
        fi
    }
    trap cleanup_guard EXIT
else
    VAL_REWARD_FUNCTION="${SCRIPT_DIR}/reward/safety_reward.py:compute_score"
    VAL_FORMAT_MODE=${VAL_FORMAT_MODE:-${FORMAT_MODE}}
fi

OUTPUT_PATH=${OUTPUT_PATH:-"${PROJECT_DIR}/checkpoints/opsd_safety_${DATA_SOURCE}"}
FIND_LAST_CHECKPOINT=${FIND_LAST_CHECKPOINT:-false}
LOAD_CHECKPOINT_PATH=${LOAD_CHECKPOINT_PATH:-}

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-1}
# Val / ARS probe dataloader batch (both use data.val_batch_size).
# Default 8 in yaml is slow on large GPUs; raise for fewer generate_sequences rounds.
# -1 = whole val/ARS set in one batch (ok when VRAM is ample).
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
# vLLM concurrency for long val/ARS gens (val_max_response_length often 4096).
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
# FSDP↔vLLM handoff: keep false by default; set OFFLOAD_PARAMS=true to free GPU
# before vLLM wake_up (avoids cumem OOM when util is high).
OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-false}
OFFLOAD_PARAMS=${OFFLOAD_PARAMS:-false}
DISTILLATION_LOSS_TYPE=${DISTILLATION_LOSS_TYPE:-topk_forward_kl}  # topk_jsd | topk_forward_kl | topk_reverse_kl | kl | jsd
DISTILLATION_TOPK_WAS_SET=${DISTILLATION_TOPK+x}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-512}
# Safe default for sampled-token PG KL: if the user switches to kl but does not
# explicitly set DISTILLATION_TOPK, disable top-k logits distillation
# (uses A=sg(logπ_T-logπ_S), L=-A*logπ_S instead).
if [ "${DISTILLATION_LOSS_TYPE}" = "kl" ] && [ -z "${DISTILLATION_TOPK_WAS_SET}" ]; then
    DISTILLATION_TOPK=null
fi

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
ACTOR_LR=${ACTOR_LR:-5e-6}
SAVE_FREQ=${SAVE_FREQ:-50}
# true → skip optimizer / lr_scheduler / rng shards (model weights + HF config only).
SAVE_MODEL_ONLY=${SAVE_MODEL_ONLY:-true}
case "${SAVE_MODEL_ONLY}" in
    true|True|TRUE|1|yes|YES) SAVE_MODEL_ONLY_HYDRA=True ;;
    *) SAVE_MODEL_ONLY_HYDRA=False ;;
esac
VAL_FREQ=${VAL_FREQ:-50}
# val_freq==0 → disable all validation (before / mid / final). See opsd_trainer.
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
case "${VAL_BEFORE_TRAIN}" in
    true|True|TRUE|1|yes|YES) VAL_BEFORE_TRAIN_HYDRA=True ;;
    *) VAL_BEFORE_TRAIN_HYDRA=False ;;
esac
if [ "${VAL_FREQ}" = "0" ]; then
    VAL_BEFORE_TRAIN_HYDRA=False
fi
MAX_STEPS=${MAX_STEPS:-200}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
VAL_MAX_RESPONSE_LENGTH=${VAL_MAX_RESPONSE_LENGTH:-4096}

# Dynamic student rollout max length by training step (3 manual phases).
# NOTE: when enabled, applies to ALL samples (harmful + benign use the same length).
ENABLE_DYNAMIC_MAX_RESP_LEN=${ENABLE_DYNAMIC_MAX_RESP_LEN:-false}
DYN_MAX_RESP_LEN_P1_END=${DYN_MAX_RESP_LEN_P1_END:-100}
DYN_MAX_RESP_LEN_P1=${DYN_MAX_RESP_LEN_P1:-256}
DYN_MAX_RESP_LEN_P2_END=${DYN_MAX_RESP_LEN_P2_END:-300}
DYN_MAX_RESP_LEN_P2=${DYN_MAX_RESP_LEN_P2:-512}
DYN_MAX_RESP_LEN_P3=${DYN_MAX_RESP_LEN_P3:-1024}
# ARS-guided rollout length (after each mid-training val; first/last skipped by default):
# Student probe uses the same max_tokens as validation (val_max_response_length),
# then teacher DSR is measured at these student prefixes over the FULL harmful probe set
# (denominator = all probe samples; no student-unsafe ∩ teacher-alone-safe filter).
# Decided length applies to ALL train rollouts (harmful + benign stay in sync).
# Results are saved under <exp>/ars_logs/step_XXXXXX.json (+ index.jsonl).
ENABLE_ADAPTIVE_ROLLOUT=${ENABLE_ADAPTIVE_ROLLOUT:-true}
# Comma-separated student prefix lengths to probe, e.g. "128,256,1024"
ARS_PREFIX_LENGTHS=${ARS_PREFIX_LENGTHS:-"128,256,1024"}
ARS_MIN_STUDENT_FAILS=${ARS_MIN_STUDENT_FAILS:-10}
ARS_TRR_THRESH=${ARS_TRR_THRESH:-0.75}
ARS_MAX_SAMPLES=${ARS_MAX_SAMPLES:-100}
ARS_MONOTONIC=${ARS_MONOTONIC:-true}
ARS_TEACHER_MAX_TOKENS=${ARS_TEACHER_MAX_TOKENS:-2048}
ARS_SKIP_FIRST_VAL=${ARS_SKIP_FIRST_VAL:-true}  # skip ARS only at step==0
ARS_SKIP_LAST_VAL=${ARS_SKIP_LAST_VAL:-true}    # skip ARS only at step==max_steps
ARS_PREFIX_LENGTHS_HYDRA="[${ARS_PREFIX_LENGTHS}]"
# ARS probe harmful source:
#   val / wjwc                       — reuse validation harmful (default: WJ+WC mix when USE_WJWC_VAL_HARMFUL)
#   wildchat100                      — WildChat plot-100
#   wildjailbreak100                 — WildJailbreak adversarial_harmful-100
#   safechain100                     — SafeChain DSR100 val harmful-100
#   safechain100_fail_qwen3-1.7b     — 100 LlamaGuard-fail SafeChain train harms (Qwen3-1.7B)
# Aliases: safechain -> safechain100;
#          wj100 / wildjailbreak -> wildjailbreak100;
#          wjwc / wildjailbreak_wildchat / wj+wc -> val (WJ+WC mix via val);
#          safechain100_fail / sc100fail / sc100fail17 -> safechain100_fail_qwen3-1.7b
# If USE_WILDCHAT100_VAL_HARMFUL=true and source=wildchat100, prefer val so student gens are reused.
# If USE_WJWC_VAL_HARMFUL=true and source=val/wjwc, ARS reuses WJWC val student rollouts (no extra file).
# Dedicated ARS files (wildchat100 / wildjailbreak100 / …) stay independent of val (extra student rollout).
ARS_HARMFUL_SOURCE=${ARS_HARMFUL_SOURCE:-val}
case "${ARS_HARMFUL_SOURCE}" in
    safechain)
        ARS_HARMFUL_SOURCE=safechain100
        ;;
    wj100|wildjailbreak)
        ARS_HARMFUL_SOURCE=wildjailbreak100
        ;;
    wjwc|wildjailbreak_wildchat|wj+wc|wj_wc)
        ARS_HARMFUL_SOURCE=val
        ;;
    safechain100_fail|sc100fail|sc100fail17|safechain100_fail_1.7b|safechain100_fail_qwen3_1_7b)
        ARS_HARMFUL_SOURCE=safechain100_fail_qwen3-1.7b
        ;;
esac
if [ "${USE_WILDCHAT100_VAL_HARMFUL}" = "true" ] && [ "${ARS_HARMFUL_SOURCE}" = "wildchat100" ]; then
    ARS_HARMFUL_SOURCE=val
fi
SAFECHAIN100_VAL_HARM=${SAFECHAIN100_VAL_HARM:-"${SCRIPT_DIR}/datasets/ars_safechain100/val_harmful.jsonl"}
SAFECHAIN100_FAIL_QWEN3_1_7B=${SAFECHAIN100_FAIL_QWEN3_1_7B:-"${SCRIPT_DIR}/datasets/ars_safechain100_fail_qwen3-1.7b/val_harmful_fail100_qwen3-1.7b.jsonl"}
WILDJAILBREAK100_VAL_HARM=${WILDJAILBREAK100_VAL_HARM:-"${SCRIPT_DIR}/datasets/ars_wildjailbreak100/val_harmful.jsonl"}
ARS_FILES=${ARS_FILES:-}
if [ -z "${ARS_FILES}" ] && [ "${ARS_HARMFUL_SOURCE}" = "wildchat100" ]; then
    if [ ! -f "${WILDCHAT100_VAL_HARM}" ]; then
        echo "[ERROR] WildChat100 harmful file missing: ${WILDCHAT100_VAL_HARM}" >&2
        exit 1
    fi
    ARS_FILES="${WILDCHAT100_VAL_HARM}"
fi
if [ -z "${ARS_FILES}" ] && [ "${ARS_HARMFUL_SOURCE}" = "wildjailbreak100" ]; then
    if [ ! -f "${WILDJAILBREAK100_VAL_HARM}" ]; then
        echo "[ERROR] WildJailbreak100 harmful file missing: ${WILDJAILBREAK100_VAL_HARM}" >&2
        exit 1
    fi
    ARS_FILES="${WILDJAILBREAK100_VAL_HARM}"
fi
if [ -z "${ARS_FILES}" ] && [ "${ARS_HARMFUL_SOURCE}" = "safechain100" ]; then
    if [ ! -f "${SAFECHAIN100_VAL_HARM}" ]; then
        echo "[ERROR] SafeChain100 harmful file missing: ${SAFECHAIN100_VAL_HARM}" >&2
        exit 1
    fi
    ARS_FILES="${SAFECHAIN100_VAL_HARM}"
fi
if [ -z "${ARS_FILES}" ] && [ "${ARS_HARMFUL_SOURCE}" = "safechain100_fail_qwen3-1.7b" ]; then
    if [ ! -f "${SAFECHAIN100_FAIL_QWEN3_1_7B}" ]; then
        echo "[ERROR] SafeChain100-fail (Qwen3-1.7B) file missing: ${SAFECHAIN100_FAIL_QWEN3_1_7B}" >&2
        exit 1
    fi
    ARS_FILES="${SAFECHAIN100_FAIL_QWEN3_1_7B}"
fi

# Teacher/student top-k vocab overlap metrics (SwanLab + overlap_logs/).
ENABLE_TOPK_OVERLAP_METRICS=${ENABLE_TOPK_OVERLAP_METRICS:-true}
OVERLAP_TOPK=${OVERLAP_TOPK:-16}

# Token filter (实现文件: examples/safety_rl/token_filter/filters.py)
# ---------------------------------------------------------------------------
# 默认按 data_type 分流:
#   harmful/safety  -> TOKEN_FILTER_MODE (taxonomy_keep)
#   benign/overreject -> TOKEN_FILTER_BENIGN_MODE (forward_kl_topn top16)
# 若 TOKEN_FILTER_BENIGN_MODE 为空，则全体样本共用 TOKEN_FILTER_MODE。
# 两段式:
#   1) TOKEN_FILTER_MODE  选出特殊集合 S
#        taxonomy_drop | taxonomy_keep | jsd_topn | forward_kl_topn
#        (兼容旧名: drop / same_only)
#   2) TOKEN_FILTER_ACTION  对 S 的处理
#        drop        — 丢掉 S，主蒸馏只训 ~S
#        sampled_kl  — S 走 sampled-token PG KL；主蒸馏训 ~S
#
# 良性默认: forward_kl_topn — 按 teacher→student forward KL 取每条序列 top-N 训练
#   (jsd_topn 作良性 mode 时也会强制按 forward KL 排序)
# TOKEN_FILTER_BENIGN_RATIO (可选): 与良性 mode 无关的最后一步重筛
#   良性可先用 taxonomy_keep / forward_kl_topn / … 得到候选 keep，
#   再 K = round(ratio × n_harmful_keep)，从良性候选中按现有顺序取前 K 个（不重排）
#   例: 有害 taxonomy keep=100, 良性 taxonomy keep=50, ratio=0.25 → 只训良性候选里前 25 个
#   空 / 未设置 = 不截断；纯良性 microbatch 无有害参照，跳过截断
#   若未设 TOKEN_FILTER_BENIGN_MODE，全体共用 TOKEN_FILTER_MODE，ratio 仍只作用在良性行
# 可选类别 (classify_token 返回值，用于 taxonomy_*)：
#   pivot         — 安全转向枢纽 (But/However 等)
#   intent        — 意图定性 / 问题 framing
#   risk_lexicon  — risk 词表且 student_token==tea_top1；作 keep 名时超集含 risk_wo_same
#   risk_wo_same  — risk 词表且 student_token≠tea_top1（去掉同词强化）
#   function      — 标点 / 功能词
#   same          — stu/tea 表面形式相同 (旧 residual_same)
#   other         — 其它语义分歧
#
# taxonomy 配对侧（student）：
#   TOKEN_FILTER_STUDENT_SOURCE=sampled  — 学生实际采样 token（默认，推荐）
#   TOKEN_FILTER_STUDENT_SOURCE=top1     — 学生 logits argmax（旧行为）
#   teacher 侧始终为 teacher top-1
#
# 默认集合 (可在 filters.py 改 TAXONOMY_*_CATEGORIES，或用下面两个环境变量覆盖):
#   taxonomy_drop 默认 drop: same,function
#   taxonomy_keep 默认 keep: pivot,intent,risk_wo_same  (= paper K; Risk↔risk_wo_same)
#   例: TOKEN_FILTER_DROP_CATEGORIES="same,function,other"
#       TOKEN_FILTER_KEEP_CATEGORIES="pivot,intent,risk_lexicon"
#       TOKEN_FILTER_KEEP_CATEGORIES="risk_wo_same"
# ---------------------------------------------------------------------------
TOKEN_FILTER=${TOKEN_FILTER:-${SAME_TOKEN_FILTER:-true}}
TOKEN_FILTER_MODE=${TOKEN_FILTER_MODE:-${SAME_TOKEN_FILTER_MODE:-taxonomy_keep}}
TOKEN_FILTER_ACTION=${TOKEN_FILTER_ACTION:-drop}
TOKEN_FILTER_TOP_N=${TOKEN_FILTER_TOP_N:-${SAME_TOKEN_FILTER_TOP_N:-16}}
TOKEN_FILTER_LOG_UPDATED=${TOKEN_FILTER_LOG_UPDATED:-${SAME_TOKEN_FILTER_LOG_UPDATED:-true}}
TOKEN_FILTER_LOG_MAX_TOKENS=${TOKEN_FILTER_LOG_MAX_TOKENS:-${SAME_TOKEN_FILTER_LOG_MAX_TOKENS:-50000}}
TOKEN_FILTER_PATH=${TOKEN_FILTER_PATH:-"examples/safety_rl/token_filter/filters.py"}
TOKEN_FILTER_CLASSIFIER=${TOKEN_FILTER_CLASSIFIER:-qwen_rubric}  # official: qwen_rubric only
# If true: teacher top-k renorm KL/JSD only on keep positions (same loss, less vocab top-k compute)
KEEP_MASK=${KEEP_MASK:-true}
TOKEN_FILTER_KEEP_CATEGORIES=${TOKEN_FILTER_KEEP_CATEGORIES:-pivot,intent,risk_wo_same}
TOKEN_FILTER_DROP_CATEGORIES=${TOKEN_FILTER_DROP_CATEGORIES:-}
TOKEN_FILTER_STUDENT_SOURCE=${TOKEN_FILTER_STUDENT_SOURCE:-top1}
# Benign-only override (empty = same mode for all samples)
TOKEN_FILTER_BENIGN_MODE=${TOKEN_FILTER_BENIGN_MODE:-}
TOKEN_FILTER_BENIGN_ACTION=${TOKEN_FILTER_BENIGN_ACTION:-drop}
TOKEN_FILTER_BENIGN_TOP_N=${TOKEN_FILTER_BENIGN_TOP_N:-16}
TOKEN_FILTER_BENIGN_RATIO=${TOKEN_FILTER_BENIGN_RATIO:-}
if [ "${TOKEN_FILTER_BENIGN_MODE}" = "none" ] || [ "${TOKEN_FILTER_BENIGN_MODE}" = "off" ]; then
    TOKEN_FILTER_BENIGN_MODE=
fi
# Compat: historical jsd_topn for benign means forward-KL top-n.
if [ "${TOKEN_FILTER_BENIGN_MODE}" = "jsd_topn" ]; then
    TOKEN_FILTER_BENIGN_MODE=forward_kl_topn
fi
if [ "${TOKEN_FILTER_BENIGN_RATIO}" = "none" ] || [ "${TOKEN_FILTER_BENIGN_RATIO}" = "off" ]; then
    TOKEN_FILTER_BENIGN_RATIO=
fi

# Periodic teacher refresh: copy student (actor) weights to frozen teacher every N steps.
# 0 = disabled (teacher stays at initial checkpoint). Requires freeze_teacher_model=true.
TEACHER_SYNC_INTERVAL=${TEACHER_SYNC_INTERVAL:-0}

# Auto experiment name tags (override with EXPERIMENT_NAME=... if needed).
MODEL_TAG=$(basename "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')
EXP_ABLATION_TAG=""
if [ "${ENABLE_THINKING}" = "true" ]; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_think"
else
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_nothink"
fi
if [ "${ENABLE_ADAPTIVE_ROLLOUT}" = "true" ]; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_arslen$(echo "${ARS_PREFIX_LENGTHS}" | tr ',' '-')_f${ARS_MIN_STUDENT_FAILS}_t${ARS_TRR_THRESH}"
    if [ "${ARS_HARMFUL_SOURCE}" = "wildchat100" ] || [ "${USE_WILDCHAT100_VAL_HARMFUL}" = "true" ]; then
        EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_wc100"
    elif [ "${ARS_HARMFUL_SOURCE}" = "safechain100" ]; then
        EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_sc100"
    fi
elif [ "${ENABLE_DYNAMIC_MAX_RESP_LEN}" = "true" ]; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_dynlen${DYN_MAX_RESP_LEN_P1_END}-${DYN_MAX_RESP_LEN_P1}_${DYN_MAX_RESP_LEN_P2_END}-${DYN_MAX_RESP_LEN_P2}_${DYN_MAX_RESP_LEN_P3}"
else
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_max${MAX_RESPONSE_LENGTH}"
fi
if [ "${USE_WJWC_VAL_HARMFUL}" = "true" ]; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_wjwc"
fi
if [ "${TEACHER_SYNC_INTERVAL}" -gt 0 ] 2>/dev/null; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_tsync${TEACHER_SYNC_INTERVAL}"
else
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_tsync0"
fi
DISTILLATION_NAME_FOR_EXP=${DISTILLATION_LOSS_TYPE}
case "${DISTILLATION_LOSS_TYPE}" in
    kl)
        if [ "${DISTILLATION_TOPK}" = "null" ]; then
            DISTILLATION_NAME_FOR_EXP="sampledtokkl"
        fi
        ;;
    topk_*)
        EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_dk${DISTILLATION_TOPK}"
        ;;
esac
if [ "${ENABLE_MATH_MIX}" = "true" ]; then
    if [ -n "${MIX_RATIOS}" ] && [ "${MIX_TOTAL}" -gt 0 ] 2>/dev/null; then
        EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_mix$(echo "${MIX_RATIOS}" | tr ':,' '__' | tr -cd '[:alnum:]_')"
    else
        EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_mix_h${HARMFUL_N}_b${BENIGN_N}_m${MATH_N}"
    fi
fi
if [ "${USE_OPSD_DYNAMIC_PI}" = "true" ]; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_opsddynpi"
fi
if [ "${USE_DSR100EASY_DATA}" = "true" ] || [[ "${DATA_DIR}" == *"_dsr100easy_"* ]] || [ "${TEACHER_PI_MODE}" = "easy" ]; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_pieasy"
elif [ "${USE_DSR100_DATA}" = "true" ] && [[ "${DATA_DIR}" == *"_dsr100_"* ]]; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_pihard"
fi
if [ "${VAL_METHOD}" = "llamaguard" ]; then
    EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_valguard"
fi
if [ "${TOKEN_FILTER}" = "true" ]; then
    if [ -n "${TOKEN_FILTER_BENIGN_MODE}" ]; then
        EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_tfh${TOKEN_FILTER_MODE}_${TOKEN_FILTER_ACTION}_tfb${TOKEN_FILTER_BENIGN_MODE}"
        if [ "${TOKEN_FILTER_BENIGN_MODE}" = "jsd_topn" ] || [ "${TOKEN_FILTER_BENIGN_MODE}" = "forward_kl_topn" ]; then
            EXP_ABLATION_TAG="${EXP_ABLATION_TAG}${TOKEN_FILTER_BENIGN_TOP_N}"
        fi
    else
        EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_tf${TOKEN_FILTER_MODE}_${TOKEN_FILTER_ACTION}"
        if [ "${TOKEN_FILTER_MODE}" = "jsd_topn" ]; then
            EXP_ABLATION_TAG="${EXP_ABLATION_TAG}${TOKEN_FILTER_TOP_N}"
        fi
    fi
    if [ -n "${TOKEN_FILTER_BENIGN_RATIO}" ]; then
        _br_tag=$(echo "${TOKEN_FILTER_BENIGN_RATIO}" | tr '.' 'p')
        EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_tfr${_br_tag}"
    fi
    if [ "${TOKEN_FILTER_MODE}" = "taxonomy_keep" ] || [ "${TOKEN_FILTER_MODE}" = "taxonomy_drop" ]; then
        if [ "${TOKEN_FILTER_STUDENT_SOURCE}" = "top1" ] || [ "${TOKEN_FILTER_STUDENT_SOURCE}" = "argmax" ]; then
            EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_stutop1"
        else
            EXP_ABLATION_TAG="${EXP_ABLATION_TAG}_stusamp"
        fi
    fi
fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"opsd_safety_${MODEL_TAG}_${DATA_SOURCE}_${DISTILLATION_NAME_FOR_EXP}${EXP_ABLATION_TAG}_fmt${FORMAT_MODE}"}
if [ -n "${RUN_SUFFIX:-}" ] && [[ "${EXPERIMENT_NAME}" != *"_${RUN_SUFFIX}" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RUN_SUFFIX}"
fi

# SwanLab: project stays fixed across runs; experiment varies (= checkpoint dir name).
SWANLAB_PROJECT=${SWANLAB_PROJECT:-opsd_safety_${DATA_SOURCE}}
# Comma-separated: file,swanlab,console,...  (swanlab skipped automatically if not installed)
LOGGER=${LOGGER:-file,swanlab}
LOGGER_HYDRA="[${LOGGER}]"

if [ "${REBUILD_DATA:-0}" == "1" ] || [ ! -f "${TRAIN_DATA}" ]; then
    if [ "${USE_OPSD_DYNAMIC_PI}" = "true" ]; then
        echo "[INFO] Building OPSD dynamic-PI dataset -> ${DATA_DIR}"
        python3 "${SCRIPT_DIR}/scripts/build_dynamic_pi_dataset.py" \
            --base_dir "${OPSD_DYNAMIC_PI_BASE_DIR}" \
            --factors_file "${OPSD_DYNAMIC_PI_FACTORS_FILE}" \
            --output_dir "${DATA_DIR}"
    elif [[ "${DATA_DIR}" == *"_dsr100"* ]]; then
        echo "[ERROR] DSR-filtered dataset missing or REBUILD_DATA=1:"
        echo "  TRAIN_DATA=${TRAIN_DATA}"
        echo "  Do NOT rebuild with prepare_safechain (会覆盖筛选集)."
        if [[ "${DATA_DIR}" == *"_dsr100easy_"* ]]; then
            echo "  Rebuild with: bash scripts/run_build_opsd_dsr100easy_dataset.sh"
        else
            echo "  Rebuild with: bash scripts/run_build_opsd_dsr100_dataset.sh"
        fi
        exit 1
    elif [ "${ENABLE_MATH_MIX}" = "true" ]; then
        echo "[INFO] Building mixed math+safety dataset -> ${DATA_DIR} (source=${DATA_SOURCE})"
        MIX_ARGS=(
            --output_dir "${DATA_DIR}"
            --data_source "${DATA_SOURCE}"
            --math_dataset "${MATH_DATASET}"
            --train_data_size "${TRAIN_DATA_SIZE}"
            --val_data_size "${VAL_DATA_SIZE}"
            --seed "${SEED}"
        )
        if [ -n "${MIX_RATIOS}" ] && [ "${MIX_TOTAL}" -gt 0 ] 2>/dev/null; then
            MIX_ARGS+=(--mix_ratios "${MIX_RATIOS}" --total "${MIX_TOTAL}")
        else
            MIX_ARGS+=(--math "${MATH_N}" --harmful "${HARMFUL_N}" --benign "${BENIGN_N}")
        fi
        if [ -n "${MATH_LOCAL_PATH}" ]; then
            MIX_ARGS+=(--math_local_path "${MATH_LOCAL_PATH}")
        fi
        if [ "${DATA_SOURCE}" = "safechain" ]; then
            MIX_ARGS+=(--safechain_dataset "${SAFECHAIN_DATASET}")
            if [ -n "${SAFECHAIN_LOCAL_PATH}" ]; then
                MIX_ARGS+=(--safechain_local_path "${SAFECHAIN_LOCAL_PATH}")
            fi
            if [ "${USE_SAFECHAIN_RESPONSE_HINT:-false}" = "true" ]; then
                MIX_ARGS+=(--use_response_hint)
            fi
        elif [ "${DATA_SOURCE}" = "star1" ]; then
            MIX_ARGS+=(--star1_harmful_dataset "${STAR1_HARMFUL_DATASET}")
            MIX_ARGS+=(--star1_benign_dataset "${STAR1_BENIGN_DATASET}")
            if [ -n "${STAR1_LOCAL_PATH}" ]; then
                MIX_ARGS+=(--star1_local_path "${STAR1_LOCAL_PATH}")
            fi
            if [ -n "${STAR1_BENIGN_LOCAL_PATH}" ]; then
                MIX_ARGS+=(--star1_benign_local_path "${STAR1_BENIGN_LOCAL_PATH}")
            fi
            if [ "${USE_SAFECHAIN_RESPONSE_HINT:-false}" = "true" ]; then
                MIX_ARGS+=(--use_response_hint)
            fi
        else
            MIX_ARGS+=(--harmful_json "${HARMFUL_JSON}" --benign_json "${BENIGN_JSON}")
        fi
        if [ "${MATH_IN_VAL}" = "true" ]; then
            MIX_ARGS+=(--math_in_val)
        fi
        python3 "${SCRIPT_DIR}/prepare_mixed_opsd_data.py" "${MIX_ARGS[@]}"
    else
        echo "[INFO] Building safety dataset -> ${DATA_DIR} (source=${DATA_SOURCE}, HF_ENDPOINT=${HF_ENDPOINT})"
        if [ "${DATA_SOURCE}" = "safechain" ]; then
            PREP_ARGS=(
                --dataset "${SAFECHAIN_DATASET}"
                --output_dir "${DATA_DIR}"
                --harmful "${HARMFUL_N}"
                --benign "${BENIGN_N}"
                --train_data_size "${TRAIN_DATA_SIZE}"
                --val_data_size "${VAL_DATA_SIZE}"
                --seed "${SEED}"
            )
            if [ -n "${SAFECHAIN_LOCAL_PATH}" ]; then
                PREP_ARGS+=(--local_path "${SAFECHAIN_LOCAL_PATH}")
            fi
            if [ "${USE_SAFECHAIN_RESPONSE_HINT:-false}" = "true" ]; then
                PREP_ARGS+=(--use_response_hint)
            fi
            python3 "${SCRIPT_DIR}/prepare_safechain_data.py" "${PREP_ARGS[@]}"
        elif [ "${DATA_SOURCE}" = "star1" ]; then
            PREP_ARGS=(
                --harmful_dataset "${STAR1_HARMFUL_DATASET}"
                --benign_dataset "${STAR1_BENIGN_DATASET}"
                --output_dir "${DATA_DIR}"
                --harmful "${HARMFUL_N}"
                --benign "${BENIGN_N}"
                --train_data_size "${TRAIN_DATA_SIZE}"
                --val_data_size "${VAL_DATA_SIZE}"
                --seed "${SEED}"
            )
            if [ -n "${STAR1_LOCAL_PATH}" ]; then
                PREP_ARGS+=(--local_path "${STAR1_LOCAL_PATH}")
            fi
            if [ -n "${STAR1_BENIGN_LOCAL_PATH}" ]; then
                PREP_ARGS+=(--benign_local_path "${STAR1_BENIGN_LOCAL_PATH}")
            fi
            if [ "${USE_SAFECHAIN_RESPONSE_HINT:-false}" = "true" ]; then
                PREP_ARGS+=(--use_response_hint)
            fi
            python3 "${SCRIPT_DIR}/prepare_star1_data.py" "${PREP_ARGS[@]}"
        else
            python3 "${SCRIPT_DIR}/prepare_safety_data.py" \
                --harmful_json "${HARMFUL_JSON}" \
                --benign_json "${BENIGN_JSON}" \
                --harmful "${HARMFUL_N}" \
                --benign "${BENIGN_N}" \
                --output_dir "${DATA_DIR}" \
                --train_data_size "${TRAIN_DATA_SIZE}" \
                --val_data_size "${VAL_DATA_SIZE}" \
                --seed "${SEED}"
        fi
    fi
fi

LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/opsd_safety_${TIMESTAMP}.log"
SAVE_CHECKPOINT_PATH="${OUTPUT_PATH}/${EXPERIMENT_NAME}"
mkdir -p "${SAVE_CHECKPOINT_PATH}"

TRAIN_SAMPLES=$(wc -l < "${TRAIN_DATA}")
VAL_SAMPLES=$(wc -l < "${VAL_DATA}")
STEPS_PER_EPOCH=$(( TRAIN_SAMPLES / ROLLOUT_BATCH_SIZE ))
if [ -n "${MAX_STEPS}" ]; then
    ESTIMATED_TOTAL_STEPS=${MAX_STEPS}
else
    ESTIMATED_TOTAL_STEPS=$(( STEPS_PER_EPOCH * TOTAL_EPOCHS ))
fi

echo "============================================================"
if [ "${ENABLE_MATH_MIX}" = "true" ]; then
    echo "  Mixed OPSD (math+safety+benign)  data_source=${DATA_SOURCE}  val=${VAL_METHOD}"
else
    echo "  Pure OPSD (SAFETY)  data_source=${DATA_SOURCE}  val=${VAL_METHOD}"
fi
echo "============================================================"
echo "  GPUs:    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  nproc=${NPROC_PER_NODE}"
echo "  Config:  ${CONFIG_PATH}"
echo "  Data:    ${DATA_DIR}"
if [ "${USE_OPSD_DYNAMIC_PI}" = "true" ]; then
    echo "  opsd_dynamic_pi=true"
    echo "  opsd_dynamic_pi_base=${OPSD_DYNAMIC_PI_BASE_DIR}"
    echo "  opsd_dynamic_pi_factors=${OPSD_DYNAMIC_PI_FACTORS_FILE}"
else
    echo "  opsd_dynamic_pi=false"
fi
echo "  Model:   ${MODEL_PATH}"
echo "  Train:   ${TRAIN_DATA} (${TRAIN_SAMPLES} samples)"
echo "  Val:     ${VAL_DATA} (${VAL_SAMPLES} samples)"
echo "  Loss:    ${DISTILLATION_LOSS_TYPE}  topk=${DISTILLATION_TOPK}  exp_loss_tag=${DISTILLATION_NAME_FOR_EXP}"
echo "  enable_thinking=${ENABLE_THINKING}  format_mode=${FORMAT_MODE}"
echo "  format_prompt=${FORMAT_PROMPT}"
echo "  teacher_template=${TEACHER_TEMPLATE}"
echo "  teacher_pi_mode=${TEACHER_PI_MODE}"
echo "  max_response_length=${MAX_RESPONSE_LENGTH}  val_max_response_length=${VAL_MAX_RESPONSE_LENGTH}"
if [ "${ENABLE_MATH_MIX}" = "true" ]; then
    if [ -n "${MIX_RATIOS}" ] && [ "${MIX_TOTAL}" -gt 0 ] 2>/dev/null; then
        echo "  mix: ratios=${MIX_RATIOS} total=${MIX_TOTAL} math_in_val=${MATH_IN_VAL}"
    else
        echo "  mix: harmful=${HARMFUL_N} benign=${BENIGN_N} math=${MATH_N} math_in_val=${MATH_IN_VAL}"
    fi
fi
if [ "${ENABLE_ADAPTIVE_ROLLOUT}" = "true" ]; then
    echo "  adaptive_rollout: ON (harmful+benign same length)  lengths=${ARS_PREFIX_LENGTHS}  min_fails=${ARS_MIN_STUDENT_FAILS}  trr_thresh=${ARS_TRR_THRESH}  max_samples=${ARS_MAX_SAMPLES}"
    echo "  ars_skip_step0/max_step=${ARS_SKIP_FIRST_VAL}/${ARS_SKIP_LAST_VAL}  (ARS iff step not in {0,max_steps})"
    echo "  ars_harmful_source=${ARS_HARMFUL_SOURCE}  adaptive_rollout_files=${ARS_FILES:-<val>}  logs=<exp>/ars_logs/"
elif [ "${ENABLE_DYNAMIC_MAX_RESP_LEN}" = "true" ]; then
    echo "  dynamic_max_resp_len: ON (harmful+benign same length)  step<=${DYN_MAX_RESP_LEN_P1_END}->${DYN_MAX_RESP_LEN_P1}, \
<=${DYN_MAX_RESP_LEN_P2_END}->${DYN_MAX_RESP_LEN_P2}, else->${DYN_MAX_RESP_LEN_P3}"
else
    echo "  dynamic_max_resp_len: disabled"
fi
echo "  use_wjwc_val_harmful=${USE_WJWC_VAL_HARMFUL}  use_wildchat100_val_harmful=${USE_WILDCHAT100_VAL_HARMFUL}"
echo "  overlap_metrics=${ENABLE_TOPK_OVERLAP_METRICS}  overlap_topk=${OVERLAP_TOPK}"
if [ -n "${TOKEN_FILTER_BENIGN_MODE}" ]; then
    echo "  token_filter=${TOKEN_FILTER}  harmful=${TOKEN_FILTER_MODE}/${TOKEN_FILTER_ACTION}  benign=${TOKEN_FILTER_BENIGN_MODE}/${TOKEN_FILTER_BENIGN_ACTION}  module=${TOKEN_FILTER_PATH}"
else
    echo "  token_filter=${TOKEN_FILTER} (all samples)  mode=${TOKEN_FILTER_MODE}  action=${TOKEN_FILTER_ACTION}  module=${TOKEN_FILTER_PATH}"
fi
echo "  token_filter_classifier=${TOKEN_FILTER_CLASSIFIER}"
if [ "${TOKEN_FILTER}" = "true" ] && [ "${TOKEN_FILTER_MODE}" = "jsd_topn" ]; then
    echo "  token_filter_top_n=${TOKEN_FILTER_TOP_N}"
fi
if [ "${TOKEN_FILTER}" = "true" ] && { [ "${TOKEN_FILTER_BENIGN_MODE}" = "jsd_topn" ] || [ "${TOKEN_FILTER_BENIGN_MODE}" = "forward_kl_topn" ]; }; then
    echo "  token_filter_benign_top_n=${TOKEN_FILTER_BENIGN_TOP_N} (ranked by forward KL)"
fi
if [ "${TOKEN_FILTER}" = "true" ] && [ -n "${TOKEN_FILTER_BENIGN_RATIO}" ]; then
    echo "  token_filter_benign_ratio=${TOKEN_FILTER_BENIGN_RATIO} (keep first K benign tokens in order; K=round(ratio×harmful keep))"
fi
if [ "${TOKEN_FILTER}" = "true" ] && [ "${TOKEN_FILTER_MODE}" = "taxonomy_keep" ] && [ -n "${TOKEN_FILTER_KEEP_CATEGORIES}" ]; then
    echo "  token_filter_keep_categories=${TOKEN_FILTER_KEEP_CATEGORIES}"
fi
if [ "${TOKEN_FILTER}" = "true" ] && [ "${TOKEN_FILTER_MODE}" = "taxonomy_drop" ] && [ -n "${TOKEN_FILTER_DROP_CATEGORIES}" ]; then
    echo "  token_filter_drop_categories=${TOKEN_FILTER_DROP_CATEGORIES}"
fi
if [ "${TOKEN_FILTER}" = "true" ] && { [ "${TOKEN_FILTER_MODE}" = "taxonomy_keep" ] || [ "${TOKEN_FILTER_MODE}" = "taxonomy_drop" ]; }; then
    echo "  token_filter_student_source=${TOKEN_FILTER_STUDENT_SOURCE} (paired with teacher top-1)"
fi
echo "  keep_mask=${KEEP_MASK}  (sparse topk/KL on keep positions only)"
if [ "${TOKEN_FILTER}" = "true" ]; then
    echo "  token_filter_log=${TOKEN_FILTER_LOG_UPDATED}  max=${TOKEN_FILTER_LOG_MAX_TOKENS}"
fi
if [ "${TEACHER_SYNC_INTERVAL}" -gt 0 ] 2>/dev/null; then
    echo "  teacher_sync_interval=${TEACHER_SYNC_INTERVAL} (refresh frozen teacher every N steps)"
else
    echo "  teacher_sync_interval=0 (teacher frozen at initial checkpoint)"
fi
echo "  rollout.n=${ROLLOUT_N}  batch=${ROLLOUT_BATCH_SIZE}  val_batch_size=${VAL_BATCH_SIZE}"
echo "  rollout.gpu_mem_util=${ROLLOUT_GPU_MEMORY_UTILIZATION}  tp=${TENSOR_PARALLEL_SIZE}  max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
echo "  offload_params=${OFFLOAD_PARAMS}  offload_optimizer=${OFFLOAD_OPTIMIZER}"
echo "  val_freq=${VAL_FREQ}  val_before_train=${VAL_BEFORE_TRAIN_HYDRA}  val_method=${VAL_METHOD}  val_format_mode=${VAL_FORMAT_MODE}"
echo "  save_freq=${SAVE_FREQ}  save_model_only=${SAVE_MODEL_ONLY_HYDRA}"
if [ "${VAL_METHOD}" = "llamaguard" ] && [ "${VAL_FREQ}" != "0" ]; then
    echo "  guard_backend=${GUARD_BACKEND}  guard_url=${GUARD_BASE_URL}"
    echo "  guard_gpu=${GUARD_GPU}  mem_util=${GUARD_GPU_MEM_UTIL}  max_model_len=${GUARD_MAX_MODEL_LEN}  overlap=${GUARD_ALLOW_TRAIN_GPU_OVERLAP}"
    echo "  guard_model=${GUARD_MODEL_PATH}  served_name=${GUARD_SERVED_NAME}"
fi
echo "  swanlab project=${SWANLAB_PROJECT}  experiment=${EXPERIMENT_NAME}  logger=${LOGGER}"
echo "  checkpoint=${SAVE_CHECKPOINT_PATH}"
echo "  find_last_checkpoint=${FIND_LAST_CHECKPOINT}"
echo "  steps/epoch=${STEPS_PER_EPOCH}  total~=${ESTIMATED_TOTAL_STEPS}"
echo "  prompt logs: ${SAVE_CHECKPOINT_PATH}/prompt_logs/"
echo "  overlap logs: ${SAVE_CHECKPOINT_PATH}/overlap_logs/"
if [ "${TOKEN_FILTER}" = "true" ] && [ "${TOKEN_FILTER_LOG_UPDATED}" = "true" ]; then
    echo "  token filter logs: ${SAVE_CHECKPOINT_PATH}/stf_drop_logs/"
fi
echo "============================================================"

# Optional: stop leftover local Ray from a previous run (set RAY_STOP_BEFORE=0 to skip).
if [ "${RAY_STOP_BEFORE:-1}" = "1" ]; then
    ray stop --force 2>/dev/null || true
    sleep 2
fi

TRAIN_ARGS=(
    config=${CONFIG_PATH}
    data.train_files=${TRAIN_DATA}
    data.val_files=${VAL_DATA}
    data.format_prompt=${FORMAT_PROMPT}
    data.trajectory_key=safe_reference
    data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.val_max_response_length=${VAL_MAX_RESPONSE_LENGTH}
    data.enable_dynamic_max_response_length=${ENABLE_DYNAMIC_MAX_RESP_LEN}
    data.dynamic_max_response_length_phase1_end=${DYN_MAX_RESP_LEN_P1_END}
    data.dynamic_max_response_length_phase1=${DYN_MAX_RESP_LEN_P1}
    data.dynamic_max_response_length_phase2_end=${DYN_MAX_RESP_LEN_P2_END}
    data.dynamic_max_response_length_phase2=${DYN_MAX_RESP_LEN_P2}
    data.dynamic_max_response_length_phase3=${DYN_MAX_RESP_LEN_P3}
    data.enable_adaptive_rollout=${ENABLE_ADAPTIVE_ROLLOUT}
    data.adaptive_rollout_prefix_lengths="${ARS_PREFIX_LENGTHS_HYDRA}"
    data.adaptive_rollout_min_student_fails=${ARS_MIN_STUDENT_FAILS}
    data.adaptive_rollout_trr_thresh=${ARS_TRR_THRESH}
    data.adaptive_rollout_max_samples=${ARS_MAX_SAMPLES}
    data.adaptive_rollout_monotonic=${ARS_MONOTONIC}
    data.adaptive_rollout_teacher_max_tokens=${ARS_TEACHER_MAX_TOKENS}
    data.adaptive_rollout_skip_first=${ARS_SKIP_FIRST_VAL}
    data.adaptive_rollout_skip_last=${ARS_SKIP_LAST_VAL}
    worker.actor.global_batch_size=${ROLLOUT_BATCH_SIZE}
    worker.rollout.n=${ROLLOUT_N}
    worker.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}
    worker.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    worker.actor.model.model_path=${MODEL_PATH}
    worker.actor.optim.lr=${ACTOR_LR}
    worker.reward.reward_function=${TRAIN_REWARD_FUNCTION}
    worker.val_reward.reward_function=${VAL_REWARD_FUNCTION}
    worker.val_reward.reward_function_kwargs.format_mode=${VAL_FORMAT_MODE}
    worker.actor.offload.offload_optimizer=${OFFLOAD_OPTIMIZER}
    worker.actor.offload.offload_params=${OFFLOAD_PARAMS}
    worker.actor.micro_batch_size_per_device_for_update=2
    opsd.enable_grpo_opsd_simple=False
    opsd.use_gt_as_hint=True
    opsd.alpha=1.0
    opsd.outcome_ppo_coef=0.0
    opsd.freeze_teacher_model=True
    opsd.rlsd_teacher_sync_interval=${TEACHER_SYNC_INTERVAL}
    opsd.teacher_hint_template=${TEACHER_TEMPLATE}
    opsd.distillation_loss_type=${DISTILLATION_LOSS_TYPE}
    opsd.distillation_topk=${DISTILLATION_TOPK}
    opsd.overlap_topk=${OVERLAP_TOPK}
    opsd.enable_topk_overlap_metrics=${ENABLE_TOPK_OVERLAP_METRICS}
    opsd.token_filter=${TOKEN_FILTER}
    opsd.keep_mask=${KEEP_MASK}
    opsd.token_filter_mode=${TOKEN_FILTER_MODE}
    opsd.token_filter_action=${TOKEN_FILTER_ACTION}
    opsd.token_filter_top_n=${TOKEN_FILTER_TOP_N}
    opsd.token_filter_log_updated=${TOKEN_FILTER_LOG_UPDATED}
    opsd.token_filter_log_max_tokens=${TOKEN_FILTER_LOG_MAX_TOKENS}
    opsd.token_filter_path=${TOKEN_FILTER_PATH}
    opsd.token_filter_classifier=${TOKEN_FILTER_CLASSIFIER}
    opsd.token_filter_student_source=${TOKEN_FILTER_STUDENT_SOURCE}
    opsd.token_filter_benign_mode=${TOKEN_FILTER_BENIGN_MODE:-null}
    opsd.token_filter_benign_action=${TOKEN_FILTER_BENIGN_ACTION:-null}
    opsd.token_filter_benign_top_n=${TOKEN_FILTER_BENIGN_TOP_N:-null}
    opsd.token_filter_benign_ratio=${TOKEN_FILTER_BENIGN_RATIO:-null}
    opsd.student_rollout_n=1
    trainer.project_name=${SWANLAB_PROJECT}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.logger=${LOGGER_HYDRA}
    trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH}
    trainer.find_last_checkpoint=${FIND_LAST_CHECKPOINT}
    trainer.n_gpus_per_node=${NPROC_PER_NODE}
    trainer.nnodes=1
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.save_model_only=${SAVE_MODEL_ONLY_HYDRA}
    trainer.val_freq=${VAL_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN_HYDRA}
)
if [ "${VAL_METHOD}" = "llamaguard" ]; then
    TRAIN_ARGS+=(
        worker.val_reward.reward_function_kwargs.guard_backend=${GUARD_BACKEND}
        worker.val_reward.reward_function_kwargs.guard_base_url=${GUARD_BASE_URL}
        worker.val_reward.reward_function_kwargs.guard_model_path=${GUARD_MODEL_PATH}
        worker.val_reward.reward_function_kwargs.guard_model_name=${GUARD_SERVED_NAME}
    )
fi
if [ -n "${MAX_STEPS}" ]; then
    TRAIN_ARGS+=(trainer.max_steps=${MAX_STEPS})
fi
if [ -n "${LOAD_CHECKPOINT_PATH}" ]; then
    TRAIN_ARGS+=(trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH})
fi
if [ -n "${ARS_FILES}" ]; then
    TRAIN_ARGS+=(data.adaptive_rollout_files=${ARS_FILES})
fi
if [ -n "${TOKEN_FILTER_KEEP_CATEGORIES}" ]; then
    TRAIN_ARGS+=(opsd.token_filter_keep_categories="[${TOKEN_FILTER_KEEP_CATEGORIES}]")
fi
if [ -n "${TOKEN_FILTER_DROP_CATEGORIES}" ]; then
    TRAIN_ARGS+=(opsd.token_filter_drop_categories="[${TOKEN_FILTER_DROP_CATEGORIES}]")
fi

python3 -m verl.trainer.opsd_main "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
