#!/usr/bin/env bash
# Download GGUF models for use with llama-cpp (GraphMERT local LLM).
# Replaces the need for models in ~/.ollama/models; those are not GGUF and cannot be used here.
#
# Usage:
#   ./download_gguf_models.sh [TARGET_DIR] [PRESET]
#
# TARGET_DIR: where to save .gguf files (default: ~/.cache/llama-cpp/models)
# PRESET: tinyllama | smollm2-360m | list
#
# Environment:
#   GRAPHMERT_GGUF_DIR  Overrides default TARGET_DIR when no args given.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="${GRAPHMERT_GGUF_DIR:-$HOME/.cache/llama-cpp/models}"
TARGET_DIR="${1:-$DEFAULT_DIR}"
PRESET="${2:-tinyllama}"

mkdir -p "$TARGET_DIR"

# Preset: repo_id | filename
case "$PRESET" in
  list)
    echo "Presets: tinyllama | smollam2-360m"
    echo "Target dir (default): $DEFAULT_DIR"
    echo "Example: $0 $DEFAULT_DIR tinyllama"
    exit 0
    ;;
  tinyllama)
    REPO_ID="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    FILENAME="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    ;;
  smollm2-360m)
    # If filename fails, check https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/tree/main
    REPO_ID="HuggingFaceTB/SmolLM2-360M-Instruct-GGUF"
    FILENAME="smollm2-360m-instruct-Q4_K_M.gguf"
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Use preset 'list' to see options." >&2
    exit 1
    ;;
esac

echo "Downloading $FILENAME from $REPO_ID into $TARGET_DIR"

if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download "$REPO_ID" "$FILENAME" --local-dir "$TARGET_DIR" --local-dir-use-symlinks False
else
  # Fallback: direct URL (HF resolve/main)
  URL="https://huggingface.co/${REPO_ID}/resolve/main/${FILENAME}"
  echo "Using curl (huggingface-cli not found)."
  curl -L -o "$TARGET_DIR/$FILENAME" "$URL"
fi

echo "Saved: $TARGET_DIR/$FILENAME"
echo "Use in Julia: LocalLLMConfig(model_path = \"$TARGET_DIR/$FILENAME\")"
