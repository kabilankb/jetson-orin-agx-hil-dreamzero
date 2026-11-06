#!/usr/bin/env bash
# DreamZero-DROID websocket policy server on one Jetson AGX Orin (tmux session "dreamzero").
# Lives in the dreamzero checkout (setup_jetson.sh copies it there).
#
#   ./run_dreamzero_server.sh            start on port 5000 (PORT=...)
#   ./run_dreamzero_server.sh --stop
#   tmux attach -t dreamzero             watch it;  log: logs/server_latest.log
#
# Loads in ~1 min, listens on :5000, ~60 s per 24-step action chunk, ~58 GB of the 61 GB at steady state.
cd "$(dirname "$0")"
if [ "${1:-}" = "--stop" ]; then tmux kill-session -t dreamzero 2>/dev/null; echo stopped; exit 0; fi
PORT=${PORT:-5000}
CKPT=${CKPT:-$HOME/workspace/dreamzero-ckpt/DreamZero-DROID}
[ -f "$CKPT/model.safetensors.index.json" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
mkdir -p logs
tmux kill-session -t dreamzero 2>/dev/null
# DZ_LOWMEM               streaming bf16 loader + text-encoder offload (dz_lowmem.py)
# DZ_SKIP_COMPONENT_LOAD  no Wan2.1 base-weight loading/downloads (already in the checkpoint)
# DZ_DIST_BACKEND=gloo    NCCL fails on the Orin iGPU; with one process the collectives are no-ops
# TORCHDYNAMO_DISABLE=1   torch.compile workers pushed memory to 60/61 GB on the first request
# ATTENTION_BACKEND=torch PyTorch SDPA (no Transformer Engine / flash-attn on Orin)
tmux new-session -d -s dreamzero "export DZ_LOWMEM=1 DZ_SKIP_COMPONENT_LOAD=1 DZ_DIST_BACKEND=gloo \
  TORCHDYNAMO_DISABLE=1 ATTENTION_BACKEND=torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 socket_test_optimized_AR.py \
  --port $PORT --enable-dit-cache --model-path $CKPT 2>&1 | tee logs/server_latest.log"
echo "started: tmux attach -t dreamzero   log: $(pwd)/logs/server_latest.log"
