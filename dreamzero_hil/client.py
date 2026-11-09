"""Websocket client for the DreamZero-DROID policy server (roboarena protocol, as in dreamzero eval_utils).

Protocol: on connect the server sends its config (msgpack). Each request is a msgpack dict of numpy
arrays with "endpoint": "infer" | "reset"; the reply is a msgpack numpy array of actions, or a str
with a server-side traceback.

Request keys (DROID, joint-position action space):
  observation/exterior_image_0_left, observation/exterior_image_1_left, observation/wrist_image_left
      uint8 (180, 320, 3)
  observation/joint_position (7,), observation/cartesian_position (6,), observation/gripper_position (1,)
  prompt (str), session_id (str)
Reply: (24, 8) = 24 steps of [7 absolute joint positions, gripper (>0.5 = close)].
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Callable

import numpy as np
import websockets.sync.client

from ._openpi import image_tools, msgpack_numpy

IMG_HW = (180, 320)


def _connect(uri: str, open_timeout: float):
    # websockets >= 13 accepts ping_interval/ping_timeout on the sync client; 12.x (Isaac Sim 6) does not.
    # Pings are relaxed because the server is busy for ~60 s per request on an Orin.
    kw = dict(compression=None, max_size=None, open_timeout=open_timeout)
    try:
        return websockets.sync.client.connect(uri, ping_interval=60, ping_timeout=600, **kw)
    except TypeError:
        return websockets.sync.client.connect(uri, **kw)


class PolicyConnection:
    # open_timeout covers only the TCP/websocket handshake: a reachable Jetson whose server is still loading
    # refuses at once, so this only bounds the wait for an unreachable host. The reply has no timeout.
    def __init__(self, host: str, port: int, open_timeout: float = 30.0):
        self.uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        try:
            self._ws = _connect(self.uri, open_timeout)
        except (OSError, TimeoutError) as e:
            raise ConnectionError(f"could not connect to the policy server at {self.uri}: {type(e).__name__}: {e}") from e
        self.server_config = msgpack_numpy.unpackb(self._ws.recv())
        self.last_timing: dict | None = None

    def _call(self, payload: dict):
        # per-stage timing of the last call: pack -> send -> wait (server + network) -> unpack
        t0 = time.perf_counter()
        data = self._packer.pack(payload)
        t1 = time.perf_counter()
        self._ws.send(data)
        t2 = time.perf_counter()
        resp = self._ws.recv()
        t3 = time.perf_counter()
        if isinstance(resp, str):
            raise RuntimeError(f"Error in inference server:\n{resp}")
        out = msgpack_numpy.unpackb(resp)
        t4 = time.perf_counter()
        self.last_timing = {"pack_s": round(t1 - t0, 4), "send_s": round(t2 - t1, 4), "wait_s": round(t3 - t2, 3),
                            "unpack_s": round(t4 - t3, 4), "req_bytes": len(data), "resp_bytes": len(resp)}
        return out

    def infer(self, obs: dict):
        return self._call({**obs, "endpoint": "infer"})

    def reset(self, info: dict | None = None):
        return self._call({**(info or {}), "endpoint": "reset"})

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass


class DreamZeroDroidClient:
    """Turns sim observations into DreamZero requests and serves the returned chunk step by step.

    open_loop_horizon: how many of the 24 returned actions to execute before querying again
    (8 = the dreamzero sim-eval default; 24 = use the whole chunk, ~3x fewer requests). Capped at the chunk length.
    while_waiting: called repeatedly while a request is in flight (e.g. to keep a GUI rendering).
    last_call: per-request record (round trip, stage timings, sizes) set after each request; the runner
    logs and clears it.
    """

    def __init__(self, host: str, port: int, open_loop_horizon: int = 8,
                 while_waiting: Callable[[], None] | None = None, log: Callable[[str], None] = print):
        self.conn = PolicyConnection(host, port)
        self.horizon = open_loop_horizon
        self.while_waiting = while_waiting
        self.log = log
        self.timings: list[float] = []
        self.last_call: dict | None = None
        self._warned_short = False
        self.reset()

    def reset(self):
        self.k = 0
        self.n_exec = 0  # actions of the current chunk that will be executed: min(horizon, chunk length)
        self.chunk = None
        self.session_id = str(uuid.uuid4())

    def _infer_async(self, req):
        box = {}

        def run():
            try:
                box["res"] = self.conn.infer(req)
            except BaseException as e:  # noqa: BLE001
                box["err"] = e

        th = threading.Thread(target=run, daemon=True)
        th.start()
        t0, last = time.time(), 0
        while th.is_alive():
            if self.while_waiting:
                self.while_waiting()
            else:
                th.join(0.1)
            el = int(time.time() - t0)
            if el // 30 > last:
                last = el // 30
                self.log(f"[client] waiting for DreamZero ... {el}s")
        if "err" in box:
            raise box["err"]
        return box["res"]

    def act(self, obs: dict, prompt: str) -> np.ndarray:
        """obs: right_image, left_image, wrist_image (HxWx3 uint8), joint_position (7,), gripper_position (1,)."""
        if self.chunk is None or self.k >= self.n_exec:
            self.k = 0
            tp = time.perf_counter()
            req = {
                "observation/exterior_image_0_left": image_tools.resize_with_pad(obs["right_image"], *IMG_HW),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(obs["left_image"], *IMG_HW),
                "observation/wrist_image_left": image_tools.resize_with_pad(obs["wrist_image"], *IMG_HW),
                "observation/joint_position": np.asarray(obs["joint_position"], dtype=np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                "observation/gripper_position": np.asarray(obs["gripper_position"], dtype=np.float64),
                "prompt": prompt,
                "session_id": self.session_id,
            }
            prep_s = time.perf_counter() - tp
            t = time.time()
            res = self._infer_async(req)
            acts = np.asarray(res["actions"] if isinstance(res, dict) else res)
            if acts.ndim != 2 or acts.shape[-1] != 8 or len(acts) == 0:
                raise ValueError(f"expected (N, 8) actions, got {acts.shape}")
            dt = time.time() - t
            self.timings.append(dt)
            if self.horizon > len(acts) and not self._warned_short:
                self.log(f"[client] WARNING: open_loop_horizon {self.horizon} > chunk length {len(acts)}; "
                         f"executing {len(acts)} per request")
                self._warned_short = True
            self.n_exec = min(self.horizon, len(acts))
            self.last_call = {"rtt_s": round(dt, 3), "prep_s": round(prep_s, 4), "chunk_len": int(len(acts)),
                              "n_exec": self.n_exec, "chunk_bytes": int(acts.nbytes), **(self.conn.last_timing or {})}
            q = np.asarray(obs["joint_position"])
            move = np.abs(acts[:, :7] - q[None]).max(axis=0)
            self.log(f"[client] chunk {acts.shape} in {dt:.1f}s | max joint move {np.round(move, 3).tolist()} rad"
                     f" | gripper {acts[:, 7].min():.2f}..{acts[:, 7].max():.2f}")
            self.chunk = acts
        a = np.array(self.chunk[self.k], dtype=np.float64)  # copy: msgpack-decoded arrays are read-only
        self.k += 1
        a[-1] = 1.0 if a[-1] > 0.5 else 0.0
        return a
