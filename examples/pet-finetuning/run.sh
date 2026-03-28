#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PRETRAINED=pet-mad-s-v1.5.0.ckpt
REPACKED=repacked.ckpt
CONFIG=train.yaml

echo "=== Step 1: Repack checkpoint with new head architecture ==="
mtt repack "$PRETRAINED" -o "$REPACKED" -c "$CONFIG"

echo ""
echo "=== Step 2: Fine-tune on band_gap data ==="
mtt train "$CONFIG"
