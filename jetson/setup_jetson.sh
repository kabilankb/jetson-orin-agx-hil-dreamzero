#!/usr/bin/env bash
# One-shot DreamZero setup on a Jetson AGX Orin 64GB (JetPack 7 / L4T R39, CUDA 13 driver).
#
#   ./setup_jetson.sh                 # code + venv + patches (+ checkpoint unless SKIP_CKPT=1)
#
# Env overrides:
#   DZ_DIR     dreamzero checkout        (default ~/workspace/dreamzero)
#   CKPT_DIR   checkpoint directory      (default ~/workspace/dreamzero-ckpt/DreamZero-DROID)
#   SKIP_CKPT  1 = skip the 46 GB checkpoint download
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DZ_DIR=${DZ_DIR:-$HOME/workspace/dreamzero}
CKPT_DIR=${CKPT_DIR:-$HOME/workspace/dreamzero-ckpt/DreamZero-DROID}
DZ_COMMIT=ab790c198fbce33503358efbbd4187ce9a89adf3
TORCH_INDEX=https://download.pytorch.org/whl/cu130
export PATH=$HOME/.local/bin:$PATH
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v tmux >/dev/null || echo "WARNING: tmux not found (sudo apt install tmux) - run_dreamzero_server.sh needs it"

echo "== dreamzero @ ${DZ_COMMIT:0:7} -> $DZ_DIR"
if [ ! -d "$DZ_DIR/.git" ]; then
  git clone https://github.com/dreamzero0/dreamzero.git "$DZ_DIR"
fi
git -C "$DZ_DIR" fetch --depth 1 origin "$DZ_COMMIT" 2>/dev/null || true
git -C "$DZ_DIR" checkout -q "$DZ_COMMIT"

echo "== venv (Python 3.12) + torch 2.14.1+cu130"
cd "$DZ_DIR"
[ -x .venv/bin/python ] || uv venv -p 3.12 .venv
for i in 1 2 3 4 5; do
  uv pip install -p .venv/bin/python torch==2.14.1 torchvision torchaudio --index-url $TORCH_INDEX && break
  echo "torch install attempt $i failed (flaky network?), retrying in 20 s"; sleep 20
done

echo "== inference dependencies"
uv pip install -p .venv/bin/python -r "$HERE/requirements-jetson.txt"
uv pip install -p .venv/bin/python --no-deps tianshou==0.5.1
uv pip install -p .venv/bin/python --no-deps -e .

echo "== Orin patches"
.venv/bin/python "$HERE/apply_patches.py" "$DZ_DIR"
install -m 755 "$HERE/run_dreamzero_server.sh" "$DZ_DIR/run_dreamzero_server.sh"

echo "== import check"
ATTENTION_BACKEND=torch .venv/bin/python - <<'EOF'
import sys, torch
sys.argv = ["x"]
import socket_test_optimized_AR  # noqa: F401
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
print("import OK |", torch.__version__, torch.cuda.get_device_name(0), "| bf16 matmul", float((x @ x).abs().mean()))
EOF

if [ "${SKIP_CKPT:-0}" != "1" ]; then
  echo "== checkpoint -> $CKPT_DIR (46 GB)"
  CKPT_DIR="$CKPT_DIR" "$HERE/download_checkpoint.sh"
fi
echo "done. start the server:  $DZ_DIR/run_dreamzero_server.sh"
