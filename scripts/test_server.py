"""Check a DreamZero server without Isaac Sim: one request with synthetic DROID observations.

  python scripts/test_server.py --host 192.168.1.17 [--requests 2]
Prints the server config, latency and the returned action chunk shape/range.
"""
import argparse
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dreamzero_hil.client import PolicyConnection  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="192.168.1.17")
ap.add_argument("--port", type=int, default=5000)
ap.add_argument("--requests", type=int, default=1)
ap.add_argument("--prompt", default="put the cube in the bowl")
a = ap.parse_args()

conn = PolicyConnection(a.host, a.port)
print("server config:", conn.server_config)
rng = np.random.default_rng(0)
sid = str(uuid.uuid4())
q = np.array([0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0])  # DROID sim start pose
for i in range(a.requests):
    obs = {
        "observation/exterior_image_0_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
        "observation/exterior_image_1_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
        "observation/joint_position": q,
        "observation/cartesian_position": np.zeros(6),
        "observation/gripper_position": np.zeros(1),
        "prompt": a.prompt,
        "session_id": sid,
    }
    t = time.time()
    act = np.asarray(conn.infer(obs))
    print(f"request {i}: {act.shape} in {time.time() - t:.1f}s, joints {act[:, :7].min():.2f}..{act[:, :7].max():.2f},"
          f" gripper {act[:, 7].min():.2f}..{act[:, 7].max():.2f}")
conn.close()
