# DreamZero hardware-in-the-loop bench: bench_2026-11-09_16-35-28

DreamZero-DROID on the Jetson Orin, DROID sim (Isaac Lab 3, headless) on the laptop, websocket over the LAN.

Setup: server load 81 s, LAN ping 9.062 ms avg / 93.908 ms max.

## 1. Time to finish each task

| scene | task | done? | time to finish (robot s) | time to finish (wall min) | calls | ran (steps / wall min) | object→target start→end (m) | closest gripper→object (m) |
|---|---|---|---|---|---|---|---|---|
| 1 | put the cube in the bowl | no | – | – | – | 64/64 / 8.4 | 0.259→0.259 | 0.340 |
| 2 | pick up the can and put it in the mug | no | – | – | – | 64/64 / 8.8 | 0.141→0.141 | 0.383 |
| 3 | put the banana in the bin | no | – | – | – | 64/64 / 8.7 | 0.318→0.320 | 0.397 |

Done = the object's centre is inside the target's radius in the table plane, not held up, on two checks 8 steps apart (a heuristic from physics state). A run capped below 450 steps cannot show a finish.

## 2. Action-chunk pipeline (per inference call)

| stage | where | mean | p50 | p95 |
|---|---|---|---|---|
| DreamZero inference | Jetson | 61.863 s | 61.699 s | 68.553 s |
| network both ways + server (de)serialisation | LAN | 1.369 s | 0.174 s | 12.044 s |
| step the sim through the executed actions + render | laptop | 1.405 s | 1.340 s | 1.513 s |
| one full cycle, call to call | total | 62.783 s | 63.265 s | 65.952 s |

- First call of a session: 76.2 s; later calls: 61.4 s (p95 64.6 s).
- Chunk use: 8 of 24 actions executed per call (33%); the rest is discarded and re-planned.
- Control rate achieved: 0.127 actions per wall second versus 15 Hz needed for real time (0.85% of real time).

DreamZero time per call on the Jetson:

| stage | mean | p95 |
|---|---|---|
| text encoder | 1.10 s | 10.44 s |
| image encoder | 1.22 s | 4.82 s |
| vae | 0.77 s | 1.03 s |
| kv cache | 5.89 s | 7.02 s |
| diffusion | 52.79 s | 56.32 s |
| model total | 61.82 s | 68.52 s |

## 3. Multi-task speed

- Tasks run: 3 (0 finished, success rate 0.00)
- Total: 25.9 wall min for 12.8 s of robot time (192 steps, 24 inference calls); real-time factor 0.0083
- Per task: 8.6 wall min; 6.96 tasks/hour, 0.0 finished tasks/hour
- Mean time to finish (finished tasks only): – robot s, – wall min
- A full 30 s episode at this speed: about 59.6 wall min

## 4. Resources

- Jetson ram_used_gb: mean 55.9, p95 59.0, max 59.0
- Jetson ram_total_gb: 61.4
- Jetson gpu_util_pct: mean 92.0, p95 99.0, max 100.0
- Jetson gpu_soc_power_w: mean 38.0, p95 41.9, max 42.4
- Jetson tj_temp_c: mean 69.1, p95 72.7, max 73.5
- Laptop gpu_util_pct: mean 71.0, max 100.0
- Laptop gpu_mem_mb: mean 6349.0, max 12205.0
- Laptop gpu_power_w: mean 64.3, max 113.9
