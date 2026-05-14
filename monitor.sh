#!/usr/bin/env bash
# Launch TensorBoard for this project's training runs.
# Usage: bash monitor.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Starting TensorBoard → http://localhost:6006"
tensorboard --logdir="$SCRIPT_DIR/runs" --port=6006