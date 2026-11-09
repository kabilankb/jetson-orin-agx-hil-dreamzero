#!/usr/bin/env bash
# Hardware-in-the-loop bench: DreamZero-DROID on the Jetson, the DROID sim (headless) on this laptop.
# Measures time to finish each task, where each inference cycle's time goes (action-chunk pipeline), and overall
# multi-task speed. Starts the server if it is not already up (and stops it again afterwards), samples Jetson
# tegrastats and the laptop GPU every second, runs each scene, then writes runs/bench_<time>/report.md + summary.json.
#   JETSON=atlas@192.168.1.17 scripts/bench.sh       # scenes 1-3, each until the task is done or 30 s of robot
#                                                     #   time (450 steps): up to ~1 h per scene at ~60 s per request
#   QUICK=1 scripts/bench.sh                          # 64 steps per scene (8 requests, ~10 min): pipeline numbers only
#   SCENES="1" EPISODES=3 scripts/bench.sh            # repeat a scene
# Env: JETSON (ssh target), DZ_DIR, PORT, SCENES, STEPS, EPISODES, HORIZON (default 8 = DreamZero's sim eval),
#      STOP_ON_SUCCESS (default 1), KEEP_SERVER=1 (leave a server it started running), ISAACLAB_PYTHON
# Requires key-based ssh to the Jetson; the bench should have the Jetson to itself (see README, Benchmark).
set -u
cd "$(dirname "$0")/.."
JET=${JETSON:-atlas@192.168.1.17}; HOST=${JET#*@}; PORT=${PORT:-5000}; DZ_DIR=${DZ_DIR:-workspace/dreamzero}
SCENES=${SCENES:-"1 2 3"}; EPISODES=${EPISODES:-1}; HORIZON=${HORIZON:-8}
if [ "${QUICK:-0}" = 1 ]; then STEPS=${STEPS:-64}; else STEPS=${STEPS:-0}; fi
STOP_FLAG=""; [ "${STOP_ON_SUCCESS:-1}" = 1 ] && STOP_FLAG="--stop-on-success"
PY=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/env_isaaclab/bin/python}
SLOG="~/$DZ_DIR/logs/server_latest.log"
OUT=runs/bench_$(date +%Y-%m-%d_%H-%M-%S); mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/bench.log"; }
jet() { ssh -o ConnectTimeout=8 -o BatchMode=yes "$JET" "$@"; }

started=0; tegra_pid=""; gpu_pid=""
cleanup() {
  [ -n "$gpu_pid" ] && kill "$gpu_pid" 2>/dev/null
  [ -n "$tegra_pid" ] && jet "kill $tegra_pid" 2>/dev/null
  jet "cat /tmp/dz_bench_tegrastats.log" > "$OUT/jetson_tegrastats.log" 2>/dev/null
  jet "tail -n +$((skip + 1)) $SLOG" > "$OUT/server.log" 2>/dev/null
  if [ "$started" = 1 ] && [ "${KEEP_SERVER:-0}" != 1 ]; then
    log "stopping the DreamZero server this bench started"; jet "~/$DZ_DIR/run_dreamzero_server.sh --stop" >/dev/null
  fi
}
trap cleanup EXIT
trap 'log "interrupted"; exit 130' INT TERM

jet true || { log "cannot ssh to $JET"; exit 1; }
skip=0
if jet "ss -ltn | grep -q ':$PORT '"; then
  log "using the DreamZero server already running on $HOST:$PORT"
  skip=$(jet "wc -l < $SLOG" 2>/dev/null || echo 0)
else
  log "starting the DreamZero server on $HOST:$PORT"
  jet "PORT=$PORT ~/$DZ_DIR/run_dreamzero_server.sh" >/dev/null; started=1
  t0=$(date +%s)
  until jet "ss -ltn | grep -q ':$PORT '"; do
    if jet "grep -qE 'Traceback|OutOfMemory' $SLOG"; then log "server failed:"; jet "tail -15 $SLOG" | tee -a "$OUT/bench.log"; exit 1; fi
    if ! jet "tmux has-session -t dreamzero 2>/dev/null || ss -ltn | grep -q ':$PORT '"; then
      log "server exited during load (likely OOM-killed):"; jet "tail -5 $SLOG" | tee -a "$OUT/bench.log"; exit 1
    fi
    sleep 10
  done
  echo "{\"server_load_s\": $(( $(date +%s) - t0 ))}" > "$OUT/server_load.json"
  log "server listening after $(( $(date +%s) - t0 ))s"
fi

# LAN baseline: plain round-trip latency to the Jetson, to compare with the network share of each call
ping -c 20 -i 0.2 -q "$HOST" 2>/dev/null | awk -F'[/ ]' '/^rtt|^round-trip/ {printf "{\"ping_ms\": {\"min\": %s, \"avg\": %s, \"max\": %s}}\n", $7, $8, $9}' \
  > "$OUT/network.json"
[ -s "$OUT/network.json" ] && log "LAN ping: $(cat "$OUT/network.json")"

tegra_pid=$(jet "rm -f /tmp/dz_bench_tegrastats.log; nohup tegrastats --interval 1000 --logfile /tmp/dz_bench_tegrastats.log >/dev/null 2>&1 & echo \$!")
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 1 \
    > "$OUT/laptop_gpu.csv" 2>/dev/null & gpu_pid=$!
fi

log "bench: scenes [$SCENES], steps=${STEPS} (0 = full episode), episodes=$EPISODES, horizon=$HORIZON, ${STOP_FLAG:-no early stop} -> $OUT"
for s in $SCENES; do
  log "scene $s ..."
  "$PY" -u scripts/run_sim.py --headless --scene "$s" --max-steps "$STEPS" --episodes "$EPISODES" \
    --open-loop-horizon "$HORIZON" --host "$HOST" --port "$PORT" --out "$OUT/scene$s" --metrics "$OUT/scene$s.jsonl" $STOP_FLAG \
    > "$OUT/scene$s.log" 2>&1
  # the runner exits 0 even on a crash (app.close() in its finally), so judge by the episode records
  n=$(grep -c '"type": "episode"' "$OUT/scene$s.jsonl" 2>/dev/null); n=${n:-0}
  log "scene $s finished: $n/$EPISODES episode(s) recorded$( [ "$n" -lt "$EPISODES" ] && echo ' (INCOMPLETE, see scene'$s'.log)')"
  if ! jet "ss -ltn | grep -q ':$PORT '"; then
    log "DreamZero server is no longer running (likely OOM-killed); stopping the bench. Last server lines:"
    jet "tail -4 $SLOG" | cut -c1-200 | tee -a "$OUT/bench.log"; break
  fi
done

cleanup; trap - EXIT
"$PY" -I scripts/bench_report.py "$OUT" && log "report: $OUT/report.md"
