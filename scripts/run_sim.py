"""DROID Franka sim (Isaac Sim 6 / Isaac Lab 3) driven by a DreamZero-DROID server on a Jetson AGX Orin.

  # GUI, policy on the Jetson
  python scripts/run_sim.py --viz kit --host 192.168.1.17 --scene 1
  # no server: scripted motion (checks the sim, cameras and joint control)
  python scripts/run_sim.py --headless --policy wave --max-steps 90

Each episode writes runs/<time>/episode_N.mp4 (the three model views side by side), the first frame
as PNG, and summary.json (per-chunk latency, steps).
"""
import argparse
import json
import sys
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dreamzero_hil._openpi import image_tools  # noqa: E402
from dreamzero_hil.client import DreamZeroDroidClient  # noqa: E402
from dreamzero_hil.droid_sim import SCENES, DroidEnvCfg  # noqa: E402


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


def keep_rendering():
    if app.is_running():
        app.update()


def main():
    env_cfg = DroidEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.set_scene(args.scene)
    env = gym.make("DROID-Lab3", cfg=env_cfg)
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
    print(f"[sim] scene {args.scene} prompt='{prompt}' steps={max_steps} out={out}", flush=True)
    summary = {"scene": args.scene, "prompt": prompt, "policy": args.policy, "host": args.host,
               "open_loop_horizon": args.open_loop_horizon, "episodes": []}
    for ep in range(args.episodes):
        frames = []
        n0 = len(client.timings)
        with torch.no_grad():
            for step in range(max_steps):
                o = extract(obs)
                a = client.act(o, prompt)
                frames.append(viz(o))
                if step == 0:
                    cv2.imwrite(str(out / f"ep{ep}_first_frame.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
                act = torch.tensor(a, dtype=torch.float32, device=env.unwrapped.device)[None]
                obs, _, term, trunc, _ = env.step(act)
                if not app.is_running() or bool(term) or bool(trunc):
                    break
        client.reset()
        video = out / f"episode_{ep}.mp4"
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        lat = client.timings[n0:]
        summary["episodes"].append({"steps": len(frames), "requests": len(lat),
                                    "latency_s_mean": float(np.mean(lat)) if lat else None, "video": str(video)})
        print(f"[sim] episode {ep}: {len(frames)} steps, {len(lat)} requests"
              + (f", {np.mean(lat):.1f}s/request" if lat else "") + f" -> {video}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
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
