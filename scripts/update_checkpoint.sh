#!/bin/bash
# Update the neural network checkpoint used by the web demo.
#
# Usage:
#   ./scripts/update_checkpoint.sh /path/to/checkpoint_dir
#
# This script will:
#   1. Convert the checkpoint to ONNX format (intention_network.onnx + decoder_only.onnx)
#   2. Copy the ONNX files to public/nn/
#   3. Increment the cache-busting version in the config

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/Users/charles/MIMIC-MJX/track-mjx/.venv/bin/python"

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 /Users/charles/MIMIC-MJX/track-mjx/model_checkpoints/251229_125346_172298"
    exit 1
fi

CHECKPOINT_PATH="$1"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "=== Updating neural network checkpoint ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

# Convert checkpoint to ONNX
echo "Step 1: Converting checkpoint to ONNX..."
$PYTHON "$SCRIPT_DIR/convert_checkpoint.py" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output "$PROJECT_DIR/public/nn/intention_network.onnx"

echo ""
echo "Step 2: Updating cache version..."
# Increment version query param in config to bust browser cache
CONFIG_FILE="$PROJECT_DIR/src/config/animals/rodent.ts"
if grep -q "intention_network.onnx?v=" "$CONFIG_FILE"; then
    # Extract current version and increment
    CURRENT_V=$(grep -o "intention_network.onnx?v=[0-9]*" "$CONFIG_FILE" | grep -o "[0-9]*$")
    NEW_V=$((CURRENT_V + 1))
    sed -i '' "s/intention_network.onnx?v=$CURRENT_V/intention_network.onnx?v=$NEW_V/" "$CONFIG_FILE"
    echo "Updated version: v=$CURRENT_V -> v=$NEW_V"
else
    echo "Warning: Could not find version string in config"
fi

echo ""
echo "=== Done! ==="
echo "Updated files:"
echo "  - public/nn/intention_network.onnx"
echo "  - public/nn/decoder_only.onnx"
echo ""
echo "Run 'npm run dev' to test the new checkpoint."
