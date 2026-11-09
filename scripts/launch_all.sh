#!/usr/bin/env bash
# Start the DreamZero server on the Jetson (if not running), wait until it listens, then open the sim GUI.
#   JETSON=atlas@192.168.1.17 scripts/launch_all.sh --scene 1
# Requires key-based ssh to the Jetson and the server set up with jetson/setup_jetson.sh.
set -uo pipefail
JETSON=${JETSON:-atlas@192.168.1.17}
DZ_DIR=${DZ_DIR:-workspace/dreamzero}   # relative to the Jetson user's home
PORT=${PORT:-5000}
HOST=${JETSON#*@}
cd "$(dirname "$0")/.."
log() { echo "[$(date +%T)] $*"; }

if ! ssh "$JETSON" "ss -ltn | grep -q ':$PORT '"; then
  log "starting DreamZero on $JETSON"
  ssh "$JETSON" "PORT=$PORT ~/$DZ_DIR/run_dreamzero_server.sh" || exit 1
fi
log "waiting for :$PORT (model load ~1 min) ..."
until ssh -o ConnectTimeout=8 "$JETSON" "ss -ltn | grep -q ':$PORT '"; do
  if ssh "$JETSON" "grep -qE 'Traceback|OutOfMemory' ~/$DZ_DIR/logs/server_latest.log"; then
    log "server failed:"; ssh "$JETSON" "tail -20 ~/$DZ_DIR/logs/server_latest.log"; exit 1
  fi
  # a kernel OOM kill (SIGKILL) leaves nothing in the log: the tmux session just ends
  if ! ssh "$JETSON" "tmux has-session -t dreamzero 2>/dev/null || ss -ltn | grep -q ':$PORT '"; then
    log "server exited without an error in its log (likely OOM-killed):"; ssh "$JETSON" "tail -5 ~/$DZ_DIR/logs/server_latest.log"; exit 1
  fi
  sleep 10
done
ssh "$JETSON" "grep 'dz_lowmem\] loaded' ~/$DZ_DIR/logs/server_latest.log | tail -1"
log "server is up, opening the sim"
JETSON_HOST=$HOST exec scripts/run_gui.sh --port "$PORT" "$@"
