#!/usr/bin/env bash
# Runs llama-pi0-bench-all across all 5 KV-cache modes and prints a summary table.
#
# Usage:
#   ./run-bench-all.sh [-n 3] [--warmup 1]
#
# Defaults to -n 3 --warmup 1 because Q3J1 modes are slow on CPU
# (each iteration is 10-90 s depending on mode).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BIN="${REPO_ROOT}/build/bin/llama-pi0-bench-all"
MODEL="${PI0_MODEL_DIR:-/home/genteki/gguf/pi0}"
IMG_DIR="${PI0_IMG_DIR:-/home/genteki/gentekis_document/research/pi0/data}"
IMAGES="${IMG_DIR}/cam_left_wrist.jpg,${IMG_DIR}/cam_right_wrist.jpg,${IMG_DIR}/cam_high.jpg"
PROMPT="${PI0_PROMPT:-pick up the red block}"

N=3
WARMUP=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)        N="$2"; shift 2 ;;
        --warmup)  WARMUP="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not found: ${BIN}" >&2
    echo "Build with: cmake --build ${REPO_ROOT}/build -j --target llama-pi0-bench-all" >&2
    exit 1
fi

MODES=(fp16 q4_0 q3j1_trivial q3j1_qjl_scalar q3j1_simd q3j1_fa q3j1_predeq)
LOG_DIR="${SCRIPT_DIR}/bench-logs"
mkdir -p "${LOG_DIR}"

declare -A AVG_MS
declare -A EXPERT_MS

echo "=== PI0 unified bench (-n ${N} --warmup ${WARMUP}) ==="
echo "Model: ${MODEL}"
echo "Images: ${IMAGES}"
echo "Prompt: ${PROMPT}"
echo

for mode in "${MODES[@]}"; do
    LOG="${LOG_DIR}/${mode}.log"
    echo "--- Running mode: ${mode} ---"
    "${BIN}" \
        -m "${MODEL}" \
        --image "${IMAGES}" \
        -p "${PROMPT}" \
        --kv-mode "${mode}" \
        -n "${N}" --warmup "${WARMUP}" \
        2>&1 | tee "${LOG}"

    # Parse the BENCH_RESULT line emitted at end of run.
    LINE="$(grep -E '^BENCH_RESULT' "${LOG}" | tail -1)"
    if [[ -z "${LINE}" ]]; then
        echo "WARNING: no BENCH_RESULT line in ${LOG}" >&2
        AVG_MS[${mode}]="?"
        EXPERT_MS[${mode}]="?"
    else
        # Anchor with a leading space so 'avg_ms=' doesn't also match 'expert_step_avg_ms='.
        AVG_MS[${mode}]="$(echo "${LINE}"    | grep -oE ' avg_ms=[0-9.]+'             | cut -d= -f2)"
        EXPERT_MS[${mode}]="$(echo "${LINE}" | grep -oE ' expert_step_avg_ms=[0-9.]+' | cut -d= -f2)"
    fi
    echo
done

# Summary table
BASE_AVG="${AVG_MS[fp16]}"
BASE_EXPERT="${EXPERT_MS[fp16]}"

printf '\n=== Summary (avg over %d iterations) ===\n\n' "${N}"
printf '%-15s | %12s | %18s | %12s | %15s\n' "Mode" "Inference ms" "Expert step ×10 ms" "Slowdown vs fp16" "Expert slowdown"
printf -- '----------------+--------------+--------------------+--------------+-----------------\n'
for mode in "${MODES[@]}"; do
    a="${AVG_MS[${mode}]}"
    e="${EXPERT_MS[${mode}]}"
    if [[ "${a}" == "?" ]]; then
        printf '%-15s | %12s | %18s | %12s | %15s\n' "${mode}" "?" "?" "?" "?"
        continue
    fi
    if [[ "${BASE_AVG}" == "?" ]]; then
        slow_inf="?"
        slow_exp="?"
    else
        slow_inf="$(awk -v a="${a}" -v b="${BASE_AVG}"    'BEGIN { printf "%.2fx", a/b }')"
        slow_exp="$(awk -v a="${e}" -v b="${BASE_EXPERT}" 'BEGIN { printf "%.2fx", a/b }')"
    fi
    printf '%-15s | %12.1f | %18.1f | %12s | %15s\n' "${mode}" "${a}" "${e}" "${slow_inf}" "${slow_exp}"
done

echo
echo "Per-mode logs in: ${LOG_DIR}/"
