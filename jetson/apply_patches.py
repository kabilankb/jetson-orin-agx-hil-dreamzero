"""Patch a dreamzero checkout (dreamzero0/dreamzero @ ab790c1) to run on one Jetson AGX Orin.

Run from anywhere:  python apply_patches.py /path/to/dreamzero
Idempotent. Every change is behind an environment variable, so the stock multi-GPU behaviour is
unchanged unless the variable is set (run_dreamzero_server.sh sets them all):

  DZ_LOWMEM=1               import dz_lowmem (streaming bf16 loader + text-encoder offload)
  DZ_SKIP_COMPONENT_LOAD=1  do not load/download Wan2.1 text/image encoder, VAE and DiT base
                            weights at construction (the DreamZero checkpoint already has them)
  DZ_DIST_BACKEND=gloo      process-group backend (NCCL fails on the Orin integrated GPU)
  ATTENTION_BACKEND=torch   PyTorch SDPA (the server previously forced Transformer Engine)
"""
import ast
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def patch_action_head(repo: Path):
    p = repo / "groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py"
    s = p.read_text()
    if "DZ_SKIP_COMPONENT_LOAD" in s:
        return "already patched"
    start = s.index("        text_enc_path = ensure_file(")
    end = s.index("        if not config.skip_component_loading:")
    block = s[start:end]
    indented = "".join("    " + l if l.strip() else l for l in block.splitlines(True))
    guard = '        if os.environ.get("DZ_SKIP_COMPONENT_LOAD") != "1":  # jetson: weights come from the DreamZero checkpoint\n'
    s = s[:start] + guard + indented + s[end:]
    s = s.replace(
        "        if not config.skip_component_loading:",
        '        if not config.skip_component_loading and os.environ.get("DZ_SKIP_COMPONENT_LOAD") != "1":',
        1,
    )
    ast.parse(s)
    p.write_text(s)
    return "patched"


def patch_server(repo: Path):
    p = repo / "socket_test_optimized_AR.py"
    s = p.read_text()
    changed = False
    if "dz_lowmem" not in s:
        s = s.replace(
            "from groot.vla.model.n1_5.sim_policy import GrootSimPolicy",
            'import os as _os\nif _os.environ.get("DZ_LOWMEM") == "1":\n'
            "    import dz_lowmem  # noqa: F401  jetson low-memory loader\n"
            "from groot.vla.model.n1_5.sim_policy import GrootSimPolicy",
            1,
        )
        changed = True
    if 'dist.init_process_group("nccl")' in s:
        s = s.replace(
            'dist.init_process_group("nccl")',
            'dist.init_process_group(os.environ.get("DZ_DIST_BACKEND", "nccl"))  # jetson: gloo (NCCL unsupported on Orin iGPU)',
            1,
        )
        changed = True
    if 'os.environ["ATTENTION_BACKEND"] = "TE"' in s:
        s = s.replace('os.environ["ATTENTION_BACKEND"] = "TE"', 'os.environ.setdefault("ATTENTION_BACKEND", "TE")', 1)
        changed = True
    ast.parse(s)
    # str.replace() does nothing when upstream moved an anchor, so check that every patch actually landed
    missing = [m for m in ("dz_lowmem", "DZ_DIST_BACKEND", 'os.environ.setdefault("ATTENTION_BACKEND"') if m not in s]
    if missing:
        sys.exit(f"{p}: could not apply {missing}; upstream changed, patch by hand")
    p.write_text(s)
    return "patched" if changed else "already patched"


def main():
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not (repo / "socket_test_optimized_AR.py").exists():
        sys.exit(f"{repo} is not a dreamzero checkout")
    print("action head:", patch_action_head(repo))
    print("server     :", patch_server(repo))
    shutil.copy(HERE / "dz_lowmem.py", repo / "dz_lowmem.py")
    print("dz_lowmem.py copied")


if __name__ == "__main__":
    main()
