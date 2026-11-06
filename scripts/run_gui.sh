#!/usr/bin/env bash
# Open the DROID sim with the Isaac Sim window.
#   scripts/run_gui.sh --scene 1                     # DreamZero on the Jetson ($JETSON_HOST, default 192.168.1.17)
#   scripts/run_gui.sh --policy wave                 # no server
# ISAACLAB_PYTHON: Python of an Isaac Sim 6 / Isaac Lab 3 install (default: conda env env_isaaclab).
cd "$(dirname "$0")/.."
PY=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/env_isaaclab/bin/python}
export DISPLAY=${DISPLAY:-:1}
exec "$PY" -u scripts/run_sim.py --viz kit --host "${JETSON_HOST:-192.168.1.17}" "$@"
