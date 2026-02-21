#!/bin/bash
# Update the neural network checkpoint used by the web demo.
#
# Usage:
#   ./scripts/update_checkpoint.sh /path/to/checkpoint_dir                # intention_network only
#   ./scripts/update_checkpoint.sh --with-decoder /path/to/checkpoint_dir  # both
#   ./scripts/update_checkpoint.sh --decoder-only /path/to/checkpoint_dir  # decoder_only only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/Users/charles/MIMIC-MJX/track-mjx/.venv/bin/python"

# Mode: "default" (intention only), "with-decoder" (both), "decoder-only"
MODE="default"

# Parse flags
while [[ "$1" == --* ]]; do
    case "$1" in
        --with-decoder)
            MODE="with-decoder"
            shift
            ;;
        --decoder-only)
            MODE="decoder-only"
            shift
            ;;
        *)
            echo "Unknown flag: $1"
            exit 1
            ;;
    esac
done

if [ -z "$1" ]; then
    echo "Usage: $0 [--with-decoder | --decoder-only] <checkpoint_path>"
    echo "Example: $0 /Users/charles/MIMIC-MJX/track-mjx/model_checkpoints/251229_125346_172298"
    echo ""
    echo "Options:"
    echo "  (default)       Only update intention_network.onnx"
    echo "  --with-decoder  Update both intention_network.onnx and decoder_only.onnx"
    echo "  --decoder-only  Only update decoder_only.onnx"
    exit 1
fi

CHECKPOINT_PATH="$1"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "=== Updating neural network checkpoint ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Mode: $MODE"
echo ""

# Build convert args
CONVERT_ARGS=()
if [ "$MODE" = "default" ]; then
    CONVERT_ARGS+=(--no-decoder)
elif [ "$MODE" = "decoder-only" ]; then
    CONVERT_ARGS+=(--decoder-only)
fi

# Convert checkpoint to ONNX
echo "Step 1: Converting checkpoint to ONNX..."
$PYTHON "$SCRIPT_DIR/convert_checkpoint.py" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output "$PROJECT_DIR/public/nn/intention_network.onnx" \
    "${CONVERT_ARGS[@]}"

echo ""
echo "Step 2: Logging checkpoint provenance..."
LOG_FILE="$PROJECT_DIR/public/nn/checkpoint.log"
echo "$(date -Iseconds) | $(basename "$CHECKPOINT_PATH") | $CHECKPOINT_PATH" >> "$LOG_FILE"
echo "Logged to $LOG_FILE"

echo ""
echo "Step 3: Updating cache versions..."
CONFIG_FILE="$PROJECT_DIR/src/config/animals/rodent.ts"

# Update intention_network.onnx version (unless decoder-only)
if [ "$MODE" != "decoder-only" ]; then
    if grep -q "intention_network.onnx?v=" "$CONFIG_FILE"; then
        CURRENT_V=$(grep -o "intention_network.onnx?v=[0-9]*" "$CONFIG_FILE" | grep -o "[0-9]*$")
        NEW_V=$((CURRENT_V + 1))
        sed -i '' "s/intention_network.onnx?v=$CURRENT_V/intention_network.onnx?v=$NEW_V/" "$CONFIG_FILE"
        echo "Updated intention_network.onnx: v=$CURRENT_V -> v=$NEW_V"
    fi
fi

# Update decoder_only.onnx version (unless default/no-decoder)
if [ "$MODE" != "default" ]; then
    if grep -q "decoder_only.onnx?v=" "$CONFIG_FILE"; then
        CURRENT_V=$(grep -o "decoder_only.onnx?v=[0-9]*" "$CONFIG_FILE" | grep -o "[0-9]*$")
        NEW_V=$((CURRENT_V + 1))
        sed -i '' "s/decoder_only.onnx?v=$CURRENT_V/decoder_only.onnx?v=$NEW_V/" "$CONFIG_FILE"
        echo "Updated decoder_only.onnx: v=$CURRENT_V -> v=$NEW_V"
    fi
fi

echo ""
echo "=== Done! ==="
echo "Updated files:"
if [ "$MODE" != "decoder-only" ]; then
    echo "  - public/nn/intention_network.onnx"
fi
if [ "$MODE" != "default" ]; then
    echo "  - public/nn/decoder_only.onnx"
fi
echo "  - public/nn/network_metadata.json"
echo ""
echo "Run 'npm run dev' to test the new checkpoint."
