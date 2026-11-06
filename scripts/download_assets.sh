#!/usr/bin/env bash
# DROID sim scenes + Franka/Robotiq USD (~90 MB) from HF dataset owhan/DROID-sim-environments -> <repo>/assets
cd "$(dirname "$0")/.."
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir "${DROID_ASSETS:-assets}"
ls "${DROID_ASSETS:-assets}"
