"""Summarise a bench run:  python bench_report.py runs/bench_<time>   ->  report.md + summary.json

Sections: task completion (time to finish), action-chunk pipeline (where each inference cycle's time goes),
multi-task throughput, Jetson resources.
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty sections are fine
out = Path(sys.argv[1])


def stats(xs, nd=3):
    xs = np.asarray([x for x in xs if x is not None], float)
    if xs.size == 0:
        return None
    return {"n": int(xs.size), "mean": round(float(xs.mean()), nd), "p50": round(float(np.percentile(xs, 50)), nd),
            "p95": round(float(np.percentile(xs, 95)), nd), "min": round(float(xs.min()), nd),
            "max": round(float(xs.max()), nd)}


def fmt(v, nd=2, unit=""):
    return "–" if v is None else f"{v:.{nd}f}{unit}"


# ---------------------------------------------------------------- records
scenes, calls_all = {}, []
for f in sorted(out.glob("scene*.jsonl"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
    recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    meta = next((r for r in recs if r["type"] == "meta"), {})
    sc = meta.get("scene", f.stem)
    calls = [r | {"scene": sc} for r in recs if r["type"] == "call"]
    # time between consecutive calls of an episode, minus the call itself = time the laptop spent executing
    for a, b in zip(calls, calls[1:]):
        if a["ep"] == b["ep"]:
            b["cycle_s"] = round(b["t"] - a["t"], 3)
            b["exec_s"] = round(b["cycle_s"] - b["rtt_s"], 3)
    calls_all += calls
    scenes[sc] = {"meta": meta, "calls": calls, "episodes": [r for r in recs if r["type"] == "episode"],
                  "tasks": [r for r in recs if r["type"] == "task"]}

# server-side time per call, in call order (one client at a time, so the k-th server line is the k-th call)
srv_lines = []
slog = out / "server.log"
if slog.exists():
    txt = slog.read_text(errors="replace")
    pt = re.compile(r"Time taken: Total ([\d.]+) seconds, Text Encoder ([\d.]+) seconds, Image Encoder ([\d.]+) "
                    r"seconds, VAE ([\d.]+) seconds, KV Cache Creation ([\d.]+) seconds, Diffusion ([\d.]+)")
    pi = re.compile(r"Inference Time: Total ([\d.]+) seconds, Transform: ([\d.]+) seconds, Model: ([\d.]+) seconds")
    stages = [dict(zip(("model_total", "text_encoder", "image_encoder", "vae", "kv_cache", "diffusion"),
                       map(float, m.groups()))) for m in pt.finditer(txt)]
    infers = [float(m.group(1)) for m in pi.finditer(txt)]
    for i, st in enumerate(stages):
        srv_lines.append(st | {"server_s": infers[i] if i < len(infers) else st["model_total"]})
matched = len(srv_lines) == len(calls_all)
for c, s in zip(calls_all, srv_lines):
    c["server_s"] = s["server_s"]
    if "wait_s" in c:
        c["net_s"] = round(c["wait_s"] - s["server_s"], 3)  # network both ways + server-side (de)serialisation
    else:
        c["net_s"] = round(c["rtt_s"] - s["server_s"], 3)

# ---------------------------------------------------------------- task completion
task_rows, total_wall, total_sim, total_steps, n_success, ep_count = [], 0.0, 0.0, 0, 0, 0
for sc, d in scenes.items():
    for e in d["episodes"]:
        ep_count += 1
        total_wall += e["wall_s"]; total_sim += e["sim_s"]; total_steps += e["steps"]
        tts = e.get("time_to_success")
        n_success += bool(e.get("success"))
        reach = min((t["d_grip_obj"] for t in d["tasks"] if t.get("ep") == e["ep"]), default=None)
        task_rows.append({"scene": sc, "prompt": d["meta"].get("prompt"), "ep": e["ep"], "success": bool(e.get("success")),
                          "time_to_success": tts, "steps": e["steps"], "sim_s": e["sim_s"], "wall_s": e["wall_s"],
                          "calls": e.get("calls", sum(1 for c in d["calls"] if c["ep"] == e["ep"])),
                          "d_start": e["start"]["d_obj_tgt_xy"], "d_end": e["end"]["d_obj_tgt_xy"],
                          "min_reach": reach, "moved": e.get("obj_moved_m"), "lift": e.get("obj_lift_m"),
                          "max_steps": d["meta"].get("max_steps")})

# ---------------------------------------------------------------- pipeline
def col(k):
    return stats([c.get(k) for c in calls_all])

first = [c for c in calls_all if "cycle_s" not in c]  # first call of each episode (new session: extra setup)
steady = [c for c in calls_all if "cycle_s" in c]
pipe = {k: col(k) for k in ("prep_s", "pack_s", "send_s", "server_s", "net_s", "unpack_s", "exec_s", "cycle_s", "rtt_s",
                            "req_bytes", "resp_bytes", "chunk_len", "n_exec", "chunk_bytes")}
pipe["rtt_first_s"] = stats([c["rtt_s"] for c in first])
pipe["rtt_steady_s"] = stats([c["rtt_s"] for c in steady])
srv_stage = {k: stats([s[k] for s in srv_lines]) for k in ("text_encoder", "image_encoder", "vae", "kv_cache", "diffusion",
                                                            "model_total")}
mean = lambda k: (pipe[k] or {}).get("mean")  # noqa: E731
n_exec, cycle = mean("n_exec"), mean("cycle_s")
step_dt = next((d["meta"].get("step_dt") for d in scenes.values() if d["meta"]), 1 / 15)
act_rate = n_exec / cycle if n_exec and cycle else None

# ---------------------------------------------------------------- throughput + resources
thru = {"tasks": ep_count, "successes": n_success,
        "success_rate": round(n_success / ep_count, 3) if ep_count else None,
        "total_wall_min": round(total_wall / 60, 1), "total_sim_s": round(total_sim, 1), "total_steps": total_steps,
        "total_calls": len(calls_all),
        "wall_min_per_task": round(total_wall / 60 / ep_count, 1) if ep_count else None,
        "tasks_per_hour": round(ep_count / (total_wall / 3600), 2) if total_wall else None,
        "successes_per_hour": round(n_success / (total_wall / 3600), 2) if total_wall else None,
        "mean_time_to_success_wall_min": stats([r["time_to_success"]["wall_s"] / 60 for r in task_rows
                                                if r["time_to_success"]], 1),
        "mean_time_to_success_sim_s": stats([r["time_to_success"]["sim_s"] for r in task_rows if r["time_to_success"]], 1),
        "actions_per_wall_s": round(act_rate, 3) if act_rate else None,
        "real_time_factor": round(total_sim / total_wall, 4) if total_wall else None,
        "full_episode_wall_min_est": round(np.ceil(450 / n_exec) * cycle / 60, 1) if n_exec and cycle else None}

jet = {}
tlog = out / "jetson_tegrastats.log"
if tlog.exists():
    t = tlog.read_text(errors="replace")
    jet = {"ram_used_gb": stats([int(a) / 1024 for a in re.findall(r"RAM (\d+)/\d+MB", t)], 1),
           "ram_total_gb": round(int(m.group(1)) / 1024, 1) if (m := re.search(r"RAM \d+/(\d+)MB", t)) else None,
           "gpu_util_pct": stats([int(x) for x in re.findall(r"GR3D_FREQ (\d+)%", t)], 0),
           "gpu_soc_power_w": stats([int(x) / 1000 for x in re.findall(r"VDD_GPU_SOC (\d+)mW", t)], 1),
           "tj_temp_c": stats([float(x) for x in re.findall(r"tj@([\d.]+)C", t)], 1)}
lap = {}
if (out / "laptop_gpu.csv").exists():
    rows = [r.split(",") for r in (out / "laptop_gpu.csv").read_text().splitlines() if r.count(",") >= 3]
    lap = {"gpu_util_pct": stats([float(r[1]) for r in rows], 0), "gpu_mem_mb": stats([float(r[2]) for r in rows], 0),
           "gpu_power_w": stats([float(r[3]) for r in rows if r[3].strip() not in ("", "[N/A]")], 1)}
extra = {}
for name in ("server_load.json", "network.json"):
    if (out / name).exists():
        extra |= json.loads((out / name).read_text())
bench_log = (out / "bench.log").read_text() if (out / "bench.log").exists() else ""

summary = {"tasks": task_rows, "pipeline": pipe, "server_stages": srv_stage, "server_calls_matched": matched,
           "throughput": thru, "jetson": jet, "laptop_gpu": lap, **extra}
(out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

# ---------------------------------------------------------------- markdown
L = [f"# DreamZero hardware-in-the-loop bench: {out.name}", "",
     "DreamZero-DROID on the Jetson Orin, DROID sim (Isaac Lab 3, headless) on the laptop, websocket over the LAN.", ""]
setup = []
if "server_load_s" in extra:
    setup.append(f"server load {extra['server_load_s']} s")
if "ping_ms" in extra:
    setup.append(f"LAN ping {extra['ping_ms']['avg']} ms avg / {extra['ping_ms']['max']} ms max")
if setup:
    L += ["Setup: " + ", ".join(setup) + ".", ""]
if "server is no longer running" in bench_log or "INCOMPLETE" in bench_log:
    L += ["**Warning: the bench did not complete every scene; see bench.log.**", ""]

L += ["## 1. Time to finish each task", "",
      "| scene | task | done? | time to finish (robot s) | time to finish (wall min) | calls | ran (steps / wall min) | "
      "object→target start→end (m) | closest gripper→object (m) |", "|" + "---|" * 9]
for r in task_rows:
    tts = r["time_to_success"]
    L.append(f"| {r['scene']} | {r['prompt']} | {'yes' if r['success'] else 'no'} | "
             f"{fmt(tts and tts['sim_s'], 1)} | {fmt(tts and tts['wall_s'] / 60, 1)} | {tts['calls'] if tts else '–'} | "
             f"{r['steps']}/{r['max_steps']} / {r['wall_s'] / 60:.1f} | {r['d_start']:.3f}→{r['d_end']:.3f} | "
             f"{fmt(r['min_reach'], 3)} |")
L += ["", "Done = the object's centre is inside the target's radius in the table plane, not held up, on two checks "
      "8 steps apart (a heuristic from physics state). A run capped below 450 steps cannot show a finish.", ""]

L += ["## 2. Action-chunk pipeline (per inference call)", "",
      "| stage | where | mean | p50 | p95 |", "|---|---|---|---|---|"]
rows = [("prep_s", "laptop", "resize 3 camera images to 180×320"), ("pack_s", "laptop", "msgpack the request"),
        ("send_s", "laptop", "write request to the socket"), ("server_s", "Jetson", "DreamZero inference"),
        ("net_s", "LAN", "network both ways + server (de)serialisation"), ("unpack_s", "laptop", "unpack the action chunk"),
        ("exec_s", "laptop", "step the sim through the executed actions + render"),
        ("cycle_s", "total", "one full cycle, call to call")]
for k, where, what in rows:
    v = pipe.get(k)
    if v:
        L.append(f"| {what} | {where} | {v['mean']:.3f} s | {v['p50']:.3f} s | {v['p95']:.3f} s |")
L += ["", f"- First call of a session: {fmt(mean('rtt_first_s'), 1)} s; later calls: {fmt(mean('rtt_steady_s'), 1)} s "
          f"(p95 {fmt((pipe['rtt_steady_s'] or {}).get('p95'), 1)} s)."]
if pipe.get("req_bytes"):
    L.append(f"- Payload: request {mean('req_bytes') / 1024:.0f} KB (3 images + state), response "
             f"{mean('resp_bytes'):.0f} bytes, action chunk {pipe['chunk_len']['mean']:.0f}×8 float32 "
             f"({mean('chunk_bytes'):.0f} bytes).")
if n_exec:
    L.append(f"- Chunk use: {n_exec:.0f} of {pipe['chunk_len']['mean']:.0f} actions executed per call "
             f"({n_exec / pipe['chunk_len']['mean']:.0%}); the rest is discarded and re-planned.")
if act_rate:
    L.append(f"- Control rate achieved: {act_rate:.3f} actions per wall second versus {1 / step_dt:.0f} Hz needed for "
             f"real time ({act_rate * step_dt:.2%} of real time).")
if not matched:
    L.append(f"- Server timings matched to calls by order only when counts agree: {len(srv_lines)} server entries vs "
             f"{len(calls_all)} calls, so the network column is approximate.")
L += ["", "DreamZero time per call on the Jetson:", "", "| stage | mean | p95 |", "|---|---|---|"]
for k, v in srv_stage.items():
    if v:
        L.append(f"| {k.replace('_', ' ')} | {v['mean']:.2f} s | {v['p95']:.2f} s |")

L += ["", "## 3. Multi-task speed", "",
      f"- Tasks run: {thru['tasks']} ({thru['successes']} finished, success rate {fmt(thru['success_rate'], 2)})",
      f"- Total: {thru['total_wall_min']} wall min for {thru['total_sim_s']} s of robot time "
      f"({thru['total_steps']} steps, {thru['total_calls']} inference calls); real-time factor {thru['real_time_factor']}",
      f"- Per task: {thru['wall_min_per_task']} wall min; {thru['tasks_per_hour']} tasks/hour, "
      f"{thru['successes_per_hour']} finished tasks/hour",
      f"- Mean time to finish (finished tasks only): "
      f"{fmt((thru['mean_time_to_success_sim_s'] or {}).get('mean'), 1)} robot s, "
      f"{fmt((thru['mean_time_to_success_wall_min'] or {}).get('mean'), 1)} wall min",
      f"- A full 30 s episode at this speed: about {thru['full_episode_wall_min_est']} wall min", ""]

L += ["## 4. Resources", ""]
for k, v in jet.items():
    if isinstance(v, dict):
        L.append(f"- Jetson {k}: mean {v['mean']}, p95 {v['p95']}, max {v['max']}")
    elif v is not None:
        L.append(f"- Jetson {k}: {v}")
for k, v in lap.items():
    if v:
        L.append(f"- Laptop {k}: mean {v['mean']}, max {v['max']}")
(out / "report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
