# jetson-orin-agx-hil-dreamzero

Hardware-in-the-loop for **NVIDIA DreamZero-DROID** with the policy running on a **Jetson AGX Orin 64GB** and a
Franka + Robotiq DROID scene in **Isaac Sim 6 / Isaac Lab 3** on a laptop.

```
 Laptop (Isaac Sim 6 / Isaac Lab 3)                        Jetson AGX Orin 64GB (JetPack 7)
 ┌───────────────────────────────────┐   websocket :5000   ┌───────────────────────────────────┐
 │ DROID scene: Franka + Robotiq 2F-85│ ── 3 cams 180x320 ─▶│ DreamZero-DROID 14B (bf16, 47 GB) │
 │ 2 external + wrist camera, 15 Hz  │    7 joints, grip   │ single GPU, low-memory loader     │
 │ dreamzero_hil.client              │ ◀─ 24 x (7 joints ──│ jetson/run_dreamzero_server.sh    │
 └───────────────────────────────────┘      + gripper)     └───────────────────────────────────┘
```

DreamZero ([dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero)) is a world-action model that
predicts video and actions jointly; its README targets 2× H100/GB200. This repo makes the DROID checkpoint run on
**one Orin** and closes the loop with a simulated DROID robot.

## Results (Jetson AGX Orin 64GB, L4T R39.2.1, torch 2.14.1+cu130)

| | |
|---|---|
| Model load | ~40 s (streams 10 shards; 2146 tensors, 0 missing) |
| Memory | 47 GB after load; text encoder offloaded after the prompt is encoded; plateaus at ~58 / 61 GB after 4 chunks (DreamZero caps its AR context at `max_chunk_size: 4`) |
| Latency | ~60 s per request → 24-step action chunk (1.6 s of robot motion at 15 Hz). First request ~70 s. |
| Reference | ~3 s on 2× H100, ~0.6 s on GB200 (DreamZero README) |

It is slow but correct: the first chunk for "put the cube in the bowl" is a smooth reach starting at the current
pose (shoulder +0.25 rad), and requests run indefinitely without running out of memory.

## What had to change for the Orin

All Jetson changes are opt-in environment variables applied by `jetson/apply_patches.py` to a pinned DreamZero
checkout (`ab790c1`); with the variables unset DreamZero behaves as upstream.

| Problem on the Orin | Fix | Switch |
|---|---|---|
| Stock loader holds the full state dict (46 GB) **and** a randomly initialised fp32 model (~92 GB) → ~140 GB peak | `dz_lowmem.py`: build with weight init disabled, cast to bf16, copy tensors one by one from the mmapped shards → 47 GB peak | `DZ_LOWMEM=1` |
| Constructor loads/downloads Wan2.1 text encoder, CLIP, VAE and the 14B DiT (~44 GB extra) although the checkpoint contains them | skip component loading | `DZ_SKIP_COMPONENT_LOAD=1` |
| OOM on the 3rd request (AR KV cache grows) | cache prompt embeddings, free the 11.4 GB umt5-xxl encoder after use, reload from the shards for a new prompt | part of `DZ_LOWMEM` (`DZ_TEXT_OFFLOAD=0` disables) |
| `NCCL error ... unhandled system error` on the first request | process group on gloo (single process → collectives are no-ops) | `DZ_DIST_BACKEND=gloo` |
| torch.compile workers pushed memory to 60/61 GB on the first request | disable dynamo | `TORCHDYNAMO_DISABLE=1` |
| No Transformer Engine / flash-attn for Orin | PyTorch SDPA (server no longer forces TE) | `ATTENTION_BACKEND=torch` |

PyTorch's cu130 aarch64 wheels warn that sm_87 is not in their list; they run on the Orin (sm_80 kernels).

Laptop side:

- **Isaac Sim 6 port** of the DROID scene from [arhanjain/sim-evals](https://github.com/arhanjain/sim-evals)
  (which needs Isaac Lab 2.2 / Isaac Sim 5): quaternions to (x, y, z, w), `ProxyArray` joint data, `PhysxCfg`,
  `mdp.image`. Same cameras, robot, gains, 15 Hz control / 120 Hz physics, 30 s episodes.
- Isaac Sim 6 ships `websockets` 12, whose sync client rejects `ping_interval` → handled in `client.py`.
- The request runs in a background thread and the Kit window keeps rendering while the Orin computes.
- openpi-client pins `numpy<2` (Isaac Sim 6 has numpy 2): its `msgpack_numpy` / `image_tools` are vendored.

## Setup

### 1. Jetson AGX Orin 64GB (JetPack 7 / L4T R39)

```bash
git clone <this repo> && cd jetson-orin-agx-hil-dreamzero/jetson
./setup_jetson.sh            # dreamzero @ab790c1 -> ~/workspace/dreamzero, venv, patches, 46 GB checkpoint
~/workspace/dreamzero/run_dreamzero_server.sh        # tmux "dreamzero", port 5000; --stop to stop
```

`DZ_DIR`, `CKPT_DIR`, `SKIP_CKPT=1` override the defaults. `download_checkpoint.sh` restarts stalled
Hugging Face downloads. Free memory first: stop other GPU services (vLLM, cuRobo, …) - the model needs ~58 GB.
If the Orin is on Wi-Fi and Ethernet in the same subnet, `jetson/ethernet_routes.sh` makes wired the default
route (Wi-Fi gave us ~2 MB/s, wired ~50 MB/s).

Check it without a simulator (from any machine):

```bash
python scripts/test_server.py --host <jetson-ip>
```

### 2. Laptop with Isaac Sim 6 / Isaac Lab 3

```bash
scripts/download_assets.sh                   # DROID scenes + Franka/Robotiq USD (~90 MB) -> assets/
$ISAACLAB_PYTHON -m pip install -e .         # optional; scripts also work from the repo
```

`ISAACLAB_PYTHON` defaults to `~/miniconda3/envs/env_isaaclab/bin/python`.

## Run

```bash
JETSON_HOST=192.168.1.17 scripts/run_gui.sh --scene 1                 # GUI, DreamZero on the Jetson
scripts/run_gui.sh --policy wave                                      # GUI, scripted motion, no server
$ISAACLAB_PYTHON scripts/run_sim.py --headless --policy hold --max-steps 60
JETSON=atlas@192.168.1.17 scripts/launch_all.sh --scene 2             # start the server over ssh, then the GUI
```

| Scene | Instruction |
|---|---|
| 1 | put the cube in the bowl |
| 2 | pick up the can and put it in the mug |
| 3 | put the banana in the bin |

Options: `--episodes N`, `--max-steps N`, `--prompt "..."`, `--open-loop-horizon N` (actions executed per
request; default 24 = whole chunk, DreamZero's own sim eval uses 8 → 3× more requests). Output in
`runs/<time>/`: `episode_N.mp4` (the three model views), first frame, `summary.json` (latency per request).

With ~60 s per request a full 30 s episode (450 steps) takes ~19 requests ≈ 20 min at horizon 24. The arm
moves for 1.6 s, then holds while the next chunk is computed.

## Layout

```
dreamzero_hil/
  client.py            DreamZero websocket client (roboarena protocol), chunk executor, UI-friendly waiting
  droid_sim/           DROID scene for Isaac Lab 3 (env.py, robot.py), gym id "DROID-Lab3"
  _openpi/             vendored openpi-client msgpack_numpy + image_tools (Apache-2.0)
scripts/
  run_sim.py           sim runner (dreamzero | hold | wave policies)
  run_gui.sh  launch_all.sh  download_assets.sh  test_server.py
jetson/
  setup_jetson.sh      clone + venv + deps + patches + checkpoint
  apply_patches.py     env-gated patches to the dreamzero checkout
  dz_lowmem.py         streaming bf16 loader + text-encoder offload
  run_dreamzero_server.sh  download_checkpoint.sh  ethernet_routes.sh  requirements-jetson.txt
docs/TROUBLESHOOTING.md
```

## Licenses

Apache-2.0 for this repo. Builds on DreamZero (Apache-2.0) and openpi-client (Apache-2.0); the DROID scene port is
derived from arhanjain/sim-evals, whose assets are downloaded, not redistributed. Model weights:
[GEAR-Dreams/DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID). See `NOTICE`.
