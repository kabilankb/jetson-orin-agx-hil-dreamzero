#!/usr/bin/env bash
# Download GEAR-Dreams/DreamZero-DROID (bf16 safetensors, ~46 GB; skips the 19 GB GB200-only TensorRT files).
# hf/xet downloads sometimes stall at 0 B/s: this watchdog restarts the download after STALL_SECS
# without progress (finished shards are kept) and cleans up stale partial files at the end.
set -uo pipefail
CKPT_DIR=${CKPT_DIR:-$HOME/workspace/dreamzero-ckpt/DreamZero-DROID}
STALL_SECS=${STALL_SECS:-150}
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$CKPT_DIR"

shards() { ls "$CKPT_DIR"/model-*-of-00010.safetensors 2>/dev/null | wc -l; }
start() {
  uvx --from huggingface_hub hf download GEAR-Dreams/DreamZero-DROID --repo-type model \
    --local-dir "$CKPT_DIR" --exclude "tensorrt/*" > "$CKPT_DIR/../download.log" 2>&1 &
  PID=$!
}

start
last=$(du -sb "$CKPT_DIR" | cut -f1); still=0
while kill -0 $PID 2>/dev/null; do
  sleep 30
  now=$(du -sb "$CKPT_DIR" | cut -f1)
  if [ "$now" = "$last" ]; then still=$((still + 30)); else still=0; fi
  last=$now
  echo "$(date +%T) $(shards)/10 shards, $((now / 1000000)) MB on disk"
  if [ $still -ge $STALL_SECS ]; then
    echo "no progress for ${STALL_SECS}s -> restarting download"
    pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null
    start; still=0
  fi
done
wait $PID; rc=$?
rm -f "$CKPT_DIR"/.cache/huggingface/download/*.incomplete
echo "exit $rc, $(shards)/10 shards, $(du -sh "$CKPT_DIR" | cut -f1)"
[ "$(shards)" = "10" ]
