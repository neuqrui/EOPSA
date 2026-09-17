#!/bin/bash
# Start a standalone Llama-Guard-3 vLLM OpenAI server on a dedicated GPU.
# Training Ray/FSDP must use a *different* CUDA_VISIBLE_DEVICES set.
#
# Example:
#   GUARD_GPU=5 GUARD_PORT=8123 bash scripts/start_llamaguard_vllm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFETY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

GUARD_MODEL_PATH=${GUARD_MODEL_PATH:-"meta-llama/Llama-Guard-3-8B"}
GUARD_GPU=${GUARD_GPU:-0}
GUARD_PORT=${GUARD_PORT:-8123}
GUARD_HOST=${GUARD_HOST:-127.0.0.1}
GUARD_GPU_MEM_UTIL=${GUARD_GPU_MEM_UTIL:-0.90}
# Cover long student rollouts (e.g. max_resp=8192) + prompt/template overhead.
# Llama-Guard-3-8B supports up to 131072 via rope_scaling.
GUARD_MAX_MODEL_LEN=${GUARD_MAX_MODEL_LEN:-32768}
GUARD_SERVED_NAME=${GUARD_SERVED_NAME:-"$(basename "${GUARD_MODEL_PATH}")"}
GUARD_LOG_DIR=${GUARD_LOG_DIR:-"${SAFETY_DIR}/logs/llamaguard_vllm"}
mkdir -p "${GUARD_LOG_DIR}"
GUARD_LOG=${GUARD_LOG:-"${GUARD_LOG_DIR}/guard_gpu${GUARD_GPU}_p${GUARD_PORT}.log"}
GUARD_PID_FILE=${GUARD_PID_FILE:-"${GUARD_LOG_DIR}/guard_gpu${GUARD_GPU}_p${GUARD_PORT}.pid"}

wait_ready() {
    local url="http://${GUARD_HOST}:${GUARD_PORT}/v1/models"
    local i
    for i in $(seq 1 180); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo "[OK] LlamaGuard vLLM ready: ${url}"
            return 0
        fi
        sleep 2
    done
    echo "[ERROR] Guard server not ready after ~6min. See ${GUARD_LOG}" >&2
    return 1
}

if curl -sf "http://${GUARD_HOST}:${GUARD_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[INFO] Guard already serving on ${GUARD_HOST}:${GUARD_PORT}"
    echo "ALREADY" > "${GUARD_PID_FILE}.status"
    exit 0
fi

echo "============================================================"
echo "  LlamaGuard vLLM  GPU=${GUARD_GPU}  port=${GUARD_PORT}"
echo "  model=${GUARD_MODEL_PATH}"
echo "  served_name=${GUARD_SERVED_NAME}"
echo "  max_model_len=${GUARD_MAX_MODEL_LEN}  mem_util=${GUARD_GPU_MEM_UTIL}"
echo "  log=${GUARD_LOG}"
echo "============================================================"

# Isolate this process to GUARD_GPU only (do not inherit training CUDA_VISIBLE_DEVICES).
CUDA_VISIBLE_DEVICES="${GUARD_GPU}" nohup vllm serve "${GUARD_MODEL_PATH}" \
    --host "${GUARD_HOST}" \
    --port "${GUARD_PORT}" \
    --served-model-name "${GUARD_SERVED_NAME}" \
    --gpu-memory-utilization "${GUARD_GPU_MEM_UTIL}" \
    --max-model-len "${GUARD_MAX_MODEL_LEN}" \
    --trust-remote-code \
    --dtype auto \
    >"${GUARD_LOG}" 2>&1 &
echo $! > "${GUARD_PID_FILE}"
echo "STARTED" > "${GUARD_PID_FILE}.status"
echo "[INFO] started pid=$(cat "${GUARD_PID_FILE}")"

wait_ready
