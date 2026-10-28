#!/usr/bin/env python3
"""The neural part of the generator the way a phone would run it: ONNX Runtime on the CPU, int8 denoiser and decoder, CLIP text tower from ONNX, no torch for the models.

Used by generate_motion.generate(runtime="mobile") and the "mobile" checkbox of viewer/generate.html, to see how long a generation takes with the on-device models. Models: data/mobile/
(scripts/generate/export_onnx.py; denoiser_int8.onnx 28.6 MB, decoder_int8.onnx 6.3 MB, clip_text.onnx 254 MB fp32: the phone would hold CLIP as fp16 or weight-only int8, which is not exported here, see
docs/MOBILE.md). Measured on this computer's CPU, not on a phone. The IK proxy, the gate and the retargeter are still the Python code (they would be native code on a phone), and the walk / run speed
constraint is not ported: walk and run prompts fall back to the desktop sampler.
"""
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
MOBILE = ROOT / "data" / "mobile"
_cache = {}


def available():
    return all((MOBILE / f).exists() for f in ("denoiser_int8.onnx", "decoder_int8.onnx", "clip_text.onnx"))


class Mobile:
    def __init__(self, state, threads=4):
        import onnxruntime as ort
        from transformers import CLIPTokenizer
        import diffusion_test as DT
        import torch
        self.c, self.mu0, self.threads = state["c"], state["mu0"], threads
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = threads
        opt.enable_cpu_mem_arena = False                                  # about 40% less RAM for the int8 models (docs/MOBILE.md)
        t0 = time.perf_counter()
        mk = lambda f: ort.InferenceSession(str(MOBILE / f), opt, providers=["CPUExecutionProvider"])
        self.den, self.dec, self.clip = mk("denoiser_int8.onnx"), mk("decoder_int8.onnx"), mk("clip_text.onnx")
        self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.ab = DT.schedule(torch, "cpu").numpy()
        self.torch = torch
        self.load_s = time.perf_counter() - t0

    def encode(self, prompt):
        """centred, normalised CLIP text embedding (1, 512), as generate_motion builds it"""
        ids = self.tok([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="np")["input_ids"].astype(np.int64)
        e = self.clip.run(None, {"ids": ids})[0]
        e = e / np.linalg.norm(e, axis=1, keepdims=True) - self.mu0
        return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)

    def sample(self, E, guidance=2.0, seed=0, steps=50):
        """DDIM with classifier-free guidance, 50 steps, two denoiser rows per step (conditional, unconditional); standardised latent (1, 256). Same noise as diffusion_test.sample."""
        x = self.torch.randn(1, 256, generator=self.torch.Generator(device="cpu").manual_seed(seed)).numpy()
        ts = self.torch.linspace(999, 0, steps).round().long().numpy()
        text, drop = np.repeat(E, 2, 0).astype(np.float32), np.array([False, True])
        for i, t in enumerate(ts):
            e = self.den.run(None, {"x": np.repeat(x, 2, 0).astype(np.float32), "t": np.array([t, t], np.int64), "text": text, "drop": drop})[0]
            eps = e[1:2] + guidance * (e[0:1] - e[1:2])
            a, ap = self.ab[t], (self.ab[ts[i + 1]] if i + 1 < len(ts) else 1.0)
            x0 = np.clip((x - np.sqrt(1 - a) * eps) / np.sqrt(a), -6, 6)
            x = np.sqrt(ap) * x0 + np.sqrt(1 - ap) * eps
        return x.astype(np.float32)

    def decode(self, z):
        """un-standardised latent (1, 256) -> (features (1, 128, 41), valid frames (1,), duration (1,)), like walk_check.decode"""
        f, a = self.dec.run(None, {"z": np.asarray(z, np.float32)})
        return f * self.c.std + self.c.mean, np.clip(np.round(a[:, 1] * 128), 8, 128).astype(int), np.exp(a[:, 0])


def get(state, threads=4):
    key = int(threads)
    if key not in _cache:
        _cache[key] = Mobile(state, key)
    return _cache[key]
