"""DROID Franka sim (Isaac Sim 6 / Isaac Lab 3) driven by a DreamZero-DROID server on a Jetson AGX Orin.

  # GUI, policy on the Jetson
  python scripts/run_sim.py --viz kit --host 192.168.1.17 --scene 1
  # no server: scripted motion (checks the sim, cameras and joint control)
  python scripts/run_sim.py --headless --policy wave --max-steps 90

Each episode writes runs/<time>/episode_N.avi (the three model views side by side, written frame by frame
so a crash keeps what was recorded), the first frame as PNG, summary.json, and metrics.jsonl: per-request
timing (round trip, stage breakdown, payload sizes), task progress every 8 steps (object -> target distance,
gripper reach) and per-episode results including time to finish the task. scripts/bench.sh builds on it.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", type=int, default=1, choices=[1, 2, 3])
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=0, help="0 = full 30 s episode (450 steps at 15 Hz)")
parser.add_argument("--policy", choices=["dreamzero", "hold", "wave"], default="dreamzero")
parser.add_argument("--host", default="192.168.1.17", help="Jetson running run_dreamzero_server.sh")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--open-loop-horizon", type=int, default=24,
                    help="actions executed per request (24 = whole chunk; dreamzero's own sim eval uses 8)")
parser.add_argument("--prompt", default=None, help="override the scene instruction")
parser.add_argument("--out", default=str(REPO / "runs"))
parser.add_argument("--metrics", default=None, help="JSONL metrics file (default: <run dir>/metrics.jsonl)")
parser.add_argument("--stop-on-success", action="store_true",
                    help="end the episode once the task is done (seen on 2 consecutive checks, 8 steps apart)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.open_loop_horizon < 1:
    parser.error("--open-loop-horizon must be >= 1")
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dreamzero_hil._openpi import image_tools  # noqa: E402
from dreamzero_hil.client import DreamZeroDroidClient  # noqa: E402
from dreamzero_hil.droid_sim import SCENES, TASKS, DroidEnvCfg  # noqa: E402
from dreamzero_hil.droid_sim.env import _t  # noqa: E402


def extract(obs):
    p = obs["policy"]
    g = lambda k: p[k][0].detach().cpu().numpy()  # noqa: E731
    return {
        "right_image": g("external_cam"),
        "left_image": g("external_cam_2"),
        "wrist_image": g("wrist_cam"),
        "joint_position": g("arm_joint_pos"),
        "gripper_position": g("gripper_pos"),
    }


class ScriptedClient:
    """hold: keep the start pose; wave: sinusoid on joints 1 and 4 + gripper toggling."""

    def __init__(self, mode):
        self.mode, self.t, self.q0, self.timings = mode, 0, None, []

    def reset(self):
        self.t, self.q0 = 0, None

    def act(self, o, prompt):
        if self.q0 is None:
            self.q0 = o["joint_position"].copy()
        q = self.q0.copy()
        if self.mode == "wave":
            q[0] += 0.4 * np.sin(self.t / 15.0)
            q[3] += 0.2 * np.sin(self.t / 20.0)
        self.t += 1
        grip = 1.0 if (self.mode == "wave" and (self.t // 45) % 2) else 0.0
        return np.concatenate([q, [grip]])


def viz(o):
    ims = [image_tools.resize_with_pad(o[k], 224, 224) for k in ("right_image", "wrist_image", "left_image")]
    return np.concatenate(ims, axis=1)


SIM = None  # set in main(); used to refresh the window without stepping physics


def keep_rendering():
    """Refresh the window while a request is in flight. Defensive guard: a bare app.update() steps physics when
    the Kit timeline drives it (Isaac Lab 2.x); on Isaac Lab 3 we measured zero joint motion during waits either
    way, since Isaac Lab's physics manager steps physics. Same pattern as isaaclab_physx isaac_rtx_renderer_utils."""
    if not app.is_running():
        time.sleep(0.05)
        return
    SIM.set_setting("/app/player/playSimulations", False)
    try:
        app.update()
    finally:
        SIM.set_setting("/app/player/playSimulations", True)


class Metrics:
    """Append-only JSONL, flushed per record so a crash keeps everything written so far."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def write(self, kind, **kw):
        self.f.write(json.dumps({"type": kind, "t": round(time.time(), 3), **kw}) + "\n")


class TaskProbe:
    """Task progress from physics state: object -> target distance, lift, gripper reach (between the fingers)."""

    def __init__(self, env, scene):
        self.scene = env.unwrapped.scene
        self.obj, self.tgt, self.radius = TASKS[scene]
        self.robot = self.scene["robot"]
        self.fingers = [self.robot.data.body_names.index(n) for n in ("left_inner_finger", "right_inner_finger")]

    def _pos(self, name):
        return _t(self.scene[name].data.root_pos_w)[0].cpu().numpy().astype(float)

    def sample(self):
        o, g = self._pos(self.obj), self._pos(self.tgt)
        f = _t(self.robot.data.body_pos_w)[0, self.fingers].cpu().numpy().astype(float).mean(axis=0)
        return {"obj": np.round(o, 4).tolist(), "tgt": np.round(g, 4).tolist(), "grip": np.round(f, 4).tolist(),
                "d_obj_tgt_xy": round(float(np.linalg.norm(o[:2] - g[:2])), 4),
                "d_grip_obj": round(float(np.linalg.norm(f - o)), 4)}

    def done(self, smp):
        """Heuristic: object centre inside the target radius in the table plane, and not held up in the air."""
        return bool(smp["d_obj_tgt_xy"] < self.radius and smp["obj"][2] < smp["tgt"][2] + 0.08)


def main():
    global SIM
    env_cfg = DroidEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.set_scene(args.scene)
    # No time-out term: the loop below ends the episode itself. With it, env.step() auto-resets on the last
    # step and returns the reset observation, so the final state (and the last video frame) would be lost.
    env_cfg.terminations.time_out = None
    env = gym.make("DROID-Lab3", cfg=env_cfg)
    SIM = env.unwrapped.sim
    prompt = args.prompt or SCENES[args.scene]

    obs, _ = env.reset()
    obs, _ = env.reset()  # second render pass so materials are loaded
    if args.policy == "dreamzero":
        print(f"[sim] connecting to ws://{args.host}:{args.port} ...", flush=True)
        client = DreamZeroDroidClient(args.host, args.port, args.open_loop_horizon,
                                      while_waiting=keep_rendering, log=lambda m: print(m, flush=True))
        print("[sim] server config:", client.conn.server_config, flush=True)
    else:
        client = ScriptedClient(args.policy)

    out = Path(args.out) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps or env.unwrapped.max_episode_length
    step_dt = env.unwrapped.step_dt
    metrics = Metrics(args.metrics or out / "metrics.jsonl")
    probe = TaskProbe(env, args.scene)
    metrics.write("meta", scene=args.scene, prompt=prompt, policy=args.policy, host=args.host, port=args.port,
                  horizon=args.open_loop_horizon, max_steps=max_steps, step_dt=step_dt, headless=bool(args.headless),
                  task=dict(zip(("object", "target", "radius"), TASKS[args.scene])), out=str(out))
    print(f"[sim] scene {args.scene} prompt='{prompt}' steps={max_steps} out={out}", flush=True)
    summary = {"scene": args.scene, "prompt": prompt, "policy": args.policy, "host": args.host,
               "open_loop_horizon": args.open_loop_horizon, "episodes": []}
    for ep in range(args.episodes):
        video = out / f"episode_{ep}.avi"  # MJPG AVI stays readable if the process dies mid-episode (mp4 does not)
        vw, n_frames, n0 = None, 0, len(client.timings)
        start = probe.sample()
        metrics.write("task", ep=ep, step=0, success=probe.done(start), **start)
        step_times, obj_z, t_ep = [], [], time.time()
        n_calls, ok_streak, first_ok, done_at = 0, 0, None, None  # done_at: (step, wall s, requests) when done
        try:
            with torch.no_grad():
                for step in range(max_steps):
                    o = extract(obs)
                    a = client.act(o, prompt)
                    if getattr(client, "last_call", None):
                        n_calls += 1
                        metrics.write("call", ep=ep, step=step, **client.last_call)
                        client.last_call = None
                    frame = cv2.cvtColor(viz(o), cv2.COLOR_RGB2BGR)
                    if vw is None:
                        h, w = frame.shape[:2]
                        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (w, h))
                        cv2.imwrite(str(out / f"ep{ep}_first_frame.png"), frame)
                    vw.write(frame)
                    n_frames += 1
                    act = torch.tensor(a, dtype=torch.float32, device=env.unwrapped.device)[None]
                    ts = time.perf_counter()
                    obs, _, term, trunc, _ = env.step(act)
                    step_times.append(time.perf_counter() - ts)
                    if (step + 1) % 8 == 0:
                        smp = probe.sample()
                        obj_z.append(smp["obj"][2])  # objects spawn ~3 cm up and settle in the first steps
                        ok = probe.done(smp)
                        metrics.write("task", ep=ep, step=step + 1, success=ok, **smp)
                        ok_streak = ok_streak + 1 if ok else 0
                        if ok_streak == 1:
                            first_ok = (step + 1, round(time.time() - t_ep, 2), n_calls)
                        if ok_streak == 2 and done_at is None:  # still true 8 steps later: not a pass-through
                            done_at = first_ok
                            print(f"[sim] task done at step {done_at[0]} ({done_at[0] * step_dt:.1f} s robot time, "
                                  f"{done_at[1]:.0f} s wall, {done_at[2]} requests)", flush=True)
                            if args.stop_on_success:
                                break
                    if not app.is_running() or bool(term) or bool(trunc):
                        break
        finally:
            if vw is not None:
                vw.release()
            if step_times:
                st = np.array(step_times) * 1000
                end = probe.sample()
                metrics.write(
                    "episode", ep=ep, steps=len(step_times), wall_s=round(time.time() - t_ep, 2),
                    sim_s=round(len(step_times) * step_dt, 3),
                    env_step_ms={"mean": round(float(st.mean()), 2), "p50": round(float(np.percentile(st, 50)), 2),
                                 "p95": round(float(np.percentile(st, 95)), 2), "max": round(float(st.max()), 2)},
                    start=start, end=end,
                    obj_moved_m=round(float(np.linalg.norm(np.array(end["obj"][:2]) - np.array(start["obj"][:2]))), 4),
                    obj_lift_m=round(max(obj_z) - obj_z[0], 4) if obj_z else None,
                    success=done_at is not None, calls=n_calls,
                    time_to_success=None if done_at is None else {
                        "step": done_at[0], "sim_s": round(done_at[0] * step_dt, 3), "wall_s": done_at[1],
                        "calls": done_at[2]},
                    success_at_end=probe.done(end), completed=len(step_times) == max_steps)
        client.reset()
        lat = client.timings[n0:]
        summary["episodes"].append({"steps": n_frames, "requests": len(lat), "success": done_at is not None,
                                    "latency_s_mean": float(np.mean(lat)) if lat else None, "video": str(video)})
        print(f"[sim] episode {ep}: {n_frames} steps, {len(lat)} requests"
              + (f", {np.mean(lat):.1f}s/request" if lat else "")
              + f", task {'done' if done_at else 'not done'} -> {video}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        if not app.is_running():
            break
        obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()  # Kit's shutdown can exit before Python prints the traceback
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()
