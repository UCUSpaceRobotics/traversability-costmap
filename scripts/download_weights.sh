#!/usr/bin/env bash
# Download the pretrained WayFASTER checkpoint without overwriting existing weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
URL="https://raw.githubusercontent.com/mathew4STAR/WayFaster-Ros2/main/workspace/checkpoints/best.ckpt"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [OUTPUT_DIR]"
  echo "Default output: ${REPO_ROOT}/pretrained/wayfaster_best.ckpt"
  exit 0
fi
if [[ $# -gt 1 || "${1:-}" == -* ]]; then
  echo "Usage: $0 [OUTPUT_DIR]" >&2
  exit 1
fi

OUTPUT_DIR="${1:-${REPO_ROOT}/pretrained}"
DESTINATION="${OUTPUT_DIR}/wayfaster_best.ckpt"

if [[ -e "$DESTINATION" || -L "$DESTINATION" ]]; then
  echo "[ERROR] Refusing to overwrite ${DESTINATION}" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] curl is required." >&2
  exit 1
fi

mkdir -p -- "$OUTPUT_DIR"
TEMP_FILE="$(mktemp "${OUTPUT_DIR}/.wayfaster_best.ckpt.XXXXXX")"
trap 'rm -f -- "$TEMP_FILE"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[DOWNLOAD] ${URL}"
curl -fL --retry 3 --output "$TEMP_FILE" "$URL"
if [[ ! -s "$TEMP_FILE" ]]; then
  echo "[ERROR] Downloaded checkpoint is empty." >&2
  exit 1
fi

# Publish atomically on the same filesystem; fail if another process created it.
ln -T -- "$TEMP_FILE" "$DESTINATION"
echo "[OK] Saved ${DESTINATION}"
