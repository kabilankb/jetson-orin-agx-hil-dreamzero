# Troubleshooting

## Jetson

| Symptom | Cause / fix |
|---|---|
| `torch.OutOfMemoryError` on request 3 | Text-encoder offload not active: start through `run_dreamzero_server.sh` (sets `DZ_LOWMEM=1`); log shows `[dz_lowmem] text encoder unloaded (11.4 GB freed)`. Stop other GPU services. |
| `NCCL error ... unhandled system error` | `DZ_DIST_BACKEND=gloo` missing. |
| Memory 60/61 GB, GPU 0 % for minutes on the first request | torch.compile running: `TORCHDYNAMO_DISABLE=1` missing. |
| Loader tries to download Wan-AI/Wan2.1-I2V-14B-480P | `DZ_SKIP_COMPONENT_LOAD=1` missing or patches not applied (`python jetson/apply_patches.py <dreamzero>`). |
| `UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7` | Harmless with torch 2.14.1+cu130 (sm_80 kernels run on sm_87). |
| Checkpoint download stuck at 0 B/s | Known hf/xet stall; `download_checkpoint.sh` restarts automatically. Each restart leaves `.incomplete` files (cleaned at the end). |
| torch install `operation timed out` | Slow link; `setup_jetson.sh` retries; `UV_HTTP_TIMEOUT=900`. Prefer Ethernet. |
| Everything slow over the network although Ethernet is plugged in | NetworkManager may not install end0 routes → `sudo jetson/ethernet_routes.sh`. |

## Laptop / sim

| Symptom | Cause / fix |
|---|---|
| Window closes right after `connecting to ...`, no error | Fixed in `client.py` (websockets 12 has no `ping_interval`). Tracebacks are now printed before Kit exits. |
| `TimeoutError: timed out during handshake` | Server still busy with a previous request (it is single-threaded during inference); the client waits up to 300 s. |
| Arm does not move for a minute | Expected: ~60 s per request on the Orin. Log shows `[client] chunk (24, 8) in ~60s | max joint move ...`. |
| No Isaac Sim window | Isaac Lab 3 needs `--viz kit` (run_gui.sh passes it) and a desktop `DISPLAY`. |
| Isaac Sim 5.x segfault (driver 595 / Blackwell laptop) | Use Isaac Sim 6 / Isaac Lab 3 - this repo targets it. |
