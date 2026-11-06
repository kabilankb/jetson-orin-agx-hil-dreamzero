"""Low-memory DreamZero loading for a single 64 GB Jetson Orin (unified memory).

Enabled by `import dz_lowmem` (done in socket_test_optimized_AR.py when DZ_LOWMEM=1).

Stock VLA.from_pretrained reads the full 46 GB state dict into RAM *and* builds a randomly
initialised fp32 model (~92 GB) before casting to bf16 -> ~140 GB peak. This replacement:
  1. builds the model with weight initialisation disabled (memory stays untouched/uncommitted),
  2. casts it to bf16 immediately (same dtype the stock eval_bf16 path ends up with),
  3. copies the checkpoint in tensor by tensor from the mmapped safetensors shards.
Peak resident memory is ~ the bf16 model size.
Component loading from the Wan2.1 base files is skipped via DZ_SKIP_COMPONENT_LOAD=1 (patched in
wan_flow_matching_action_tf.py): the DreamZero checkpoint already contains those weights.
"""
import contextlib
import gc
import json
import os
import time

import torch
from safetensors import safe_open

from groot.vla.model.dreamzero import base_vla

_INIT_FNS = [
    "uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_", "eye_", "dirac_",
    "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_",
]


@contextlib.contextmanager
def no_init():
    saved = {n: getattr(torch.nn.init, n) for n in _INIT_FNS if hasattr(torch.nn.init, n)}
    saved_reset = {}
    for cls in (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Embedding,
                torch.nn.LayerNorm, torch.nn.GroupNorm):
        if hasattr(cls, "reset_parameters"):
            saved_reset[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self, *a, **k: None
    for n in saved:
        setattr(torch.nn.init, n, lambda t, *a, **k: t)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)
        for cls, f in saved_reset.items():
            cls.reset_parameters = f


def _rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    return -1


@classmethod
def from_pretrained_lowmem(cls, pretrained_model_name_or_path, config=None):
    t0 = time.time()
    path = pretrained_model_name_or_path
    with open(os.path.join(path, "config.json")) as f:
        config = base_vla.VLAConfig(**json.load(f))
    ah = config.action_head_cfg
    inner = ah["config"] if "config" in ah and isinstance(ah["config"], dict) else ah
    inner["defer_lora_injection"] = False
    inner["skip_component_loading"] = True

    print("[dz_lowmem] building model without weight init ...", flush=True)
    with no_init():
        model = cls(config)
    model.to(dtype=torch.bfloat16)
    gc.collect()
    print(f"[dz_lowmem] model built + cast to bf16 in {time.time() - t0:.0f}s, rss={_rss_gb():.1f} GB", flush=True)

    target = model.state_dict(keep_vars=True)
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    loaded = set()
    unexpected = []
    for s in shards:
        with safe_open(os.path.join(path, s), framework="pt", device="cpu") as f:
            for k in f.keys():
                kk = k.replace(".base_layer.", ".")
                if kk not in target:
                    unexpected.append(k)
                    continue
                dst = target[kk]
                src = f.get_tensor(k)
                with torch.no_grad():
                    dst.data.copy_(src.to(dst.dtype) if dst.dtype != src.dtype else src)
                loaded.add(kk)
                del src
        gc.collect()
        print(f"[dz_lowmem] {s} done, rss={_rss_gb():.1f} GB, {time.time() - t0:.0f}s", flush=True)
    missing = [k for k in target if k not in loaded]
    print(f"[dz_lowmem] loaded {len(loaded)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    if missing:
        print("[dz_lowmem] missing (first 20):", missing[:20], flush=True)
    if unexpected:
        print("[dz_lowmem] unexpected (first 20):", unexpected[:20], flush=True)
    return model


base_vla.VLA.from_pretrained = from_pretrained_lowmem
print("[dz_lowmem] VLA.from_pretrained patched", flush=True)


# ---------------------------------------------------------------------------------------------
# Text-encoder offload: the umt5-xxl encoder is 11.4 GB but only turns the prompt into an
# embedding. Cache embeddings per prompt and free the encoder weights after use; reload them from
# the checkpoint shards if an unseen prompt arrives. Frees memory for the growing AR KV cache.
# ---------------------------------------------------------------------------------------------
import hashlib

from groot.vla.model.dreamzero.action_head import wan_flow_matching_action_tf as _wf

_CKPT_PATH = {"path": None}
_orig_from_pretrained = from_pretrained_lowmem.__func__


@classmethod
def _from_pretrained_remember(cls, pretrained_model_name_or_path, config=None):
    _CKPT_PATH["path"] = pretrained_model_name_or_path
    return _orig_from_pretrained(cls, pretrained_model_name_or_path, config)


base_vla.VLA.from_pretrained = _from_pretrained_remember

_EMB_CACHE = {}
_PREFIX = "action_head.text_encoder."


def _te_unload(te):
    n = 0
    for p in te.parameters():
        n += p.numel()
        p.data = torch.empty(0, dtype=p.dtype, device=p.device)
    te._dz_unloaded = True
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[dz_lowmem] text encoder unloaded ({n * 2 / 1e9:.1f} GB freed)", flush=True)


def _te_reload(te, device):
    t0 = time.time()
    path = _CKPT_PATH["path"]
    params = dict(te.named_parameters())
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    shards = sorted({s for k, s in wm.items() if k.startswith(_PREFIX)})
    for s in shards:
        with safe_open(os.path.join(path, s), framework="pt", device="cpu") as f:
            for k in f.keys():
                if not k.startswith(_PREFIX):
                    continue
                name = k[len(_PREFIX):].replace(".base_layer.", ".")
                if name in params:
                    params[name].data = f.get_tensor(k).to(device=device, dtype=torch.bfloat16)
    te._dz_unloaded = False
    print(f"[dz_lowmem] text encoder reloaded in {time.time() - t0:.1f}s", flush=True)


_orig_encode_prompt = _wf.WANPolicyHead.encode_prompt


def _encode_prompt_cached(self, input_ids, attention_mask):
    h = hashlib.sha1()
    h.update(input_ids.detach().cpu().numpy().tobytes())
    h.update(attention_mask.detach().cpu().numpy().tobytes())
    key = h.hexdigest()
    if key in _EMB_CACHE:
        return _EMB_CACHE[key].clone()
    te = self.text_encoder
    if getattr(te, "_dz_unloaded", False):
        _te_reload(te, input_ids.device)
    emb = _orig_encode_prompt(self, input_ids, attention_mask)
    _EMB_CACHE[key] = emb.detach().clone()
    print(f"[dz_lowmem] cached prompt embedding {tuple(emb.shape)} ({len(_EMB_CACHE)} cached)", flush=True)
    if os.environ.get("DZ_TEXT_OFFLOAD", "1") == "1":
        _te_unload(te)
    return emb


_wf.WANPolicyHead.encode_prompt = _encode_prompt_cached
print("[dz_lowmem] text-encoder offload enabled", flush=True)
