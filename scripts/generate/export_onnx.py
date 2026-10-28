#!/usr/bin/env python3
"""Export the three neural parts of the generator to ONNX and check them against torch (the on-device audit, docs/MOBILE.md).

  python3 scripts/generate/export_onnx.py [--out data/mobile]        # needs: pip install onnx onnxruntime onnxscript   (not in environment.yml; MARIONETTE_PYLIBS=<dir> adds a pip --target folder)

  denoiser.onnx   x (B,256) latent, t (B,) step, text (B,512) centred CLIP embedding, drop (B,) bool (true = unconditional) -> eps (B,256)     28.5 M parameters
  decoder.onnx    z (B,256) un-standardised latent -> feats (B,128,41) standardised, aux (B,2) (log duration, valid fraction)                     3.5 M parameters
  clip_text.onnx  ids (B,77) int64 CLIP token ids -> text_embeds (B,512), unnormalised                                                              63.4 M parameters
Each is checked against torch in EVAL mode (the exporter leaves a wrapped model in training mode, so a naive comparison runs with dropout on and looks wrong). The denoiser and decoder are also
written as dynamic-int8 (QInt8 weights): they hold up. CLIP text is NOT: dynamic int8 also quantizes the activations and the centred embedding falls to cosine 0.36 against fp32. For CLIP use fp16
weights (lossless) or weight-only int8 (centred cosine 0.996, generated motion within noise); see docs/MOBILE.md, where the quantization experiments are described.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
if os.environ.get("MARIONETTE_PYLIBS"):
    sys.path.append(os.environ["MARIONETTE_PYLIBS"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "mobile"))
    a = ap.parse_args()
    import torch
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import generate_motion as G
    from embed import load_text_encoder
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = G._load()["c"]
    den, ae = c.model.cpu().eval(), c.ae.cpu().eval()
    _, clip_t, _ = load_text_encoder()
    clip_t = clip_t.cpu().eval()

    class Wrap(torch.nn.Module):
        def __init__(self, fn, *mods):
            super().__init__()
            self.fn, self.mods = fn, torch.nn.ModuleList(mods)

        def forward(self, *x):
            return self.fn(*x)

    jobs = [("denoiser", Wrap(lambda x, t, text, drop: den(x, t, text, drop), den), (torch.randn(2, 256), torch.tensor([500, 500]), torch.randn(2, 512), torch.tensor([False, True])),
             ["x", "t", "text", "drop"], ["eps"], True),
            ("decoder", Wrap(lambda z: (ae.decode(z), ae.aux(z)), ae), (torch.randn(1, 256),), ["z"], ["feats", "aux"], True),
            ("clip_text", Wrap(lambda ids: clip_t(input_ids=ids).text_embeds, clip_t), (torch.randint(0, 49000, (1, 77)),), ["ids"], ["text_embeds"], False)]
    for name, mod, args, ins, outs, int8 in jobs:
        path = out / f"{name}.onnx"
        dyn = {k: {0: "b"} for k in ins + outs}
        mod.eval()
        torch.onnx.export(mod, args, str(path), input_names=ins, output_names=outs, dynamic_axes=dyn, opset_version=17, dynamo=False)
        mod.eval()                                                                  # the exporter restores the wrapper's training flag
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = mod(*args)
        ref = [r.numpy() for r in (ref if isinstance(ref, tuple) else (ref,))]
        got = sess.run(None, {n: x.numpy() for n, x in zip(ins, args)})
        err = max(float(np.abs(g - r).max()) for g, r in zip(got, ref))
        line = f"{name:10s} {path.stat().st_size / 1e6:6.1f} MB  max abs error vs torch (eval mode) {err:.1e}"
        if int8:
            q = out / f"{name}_int8.onnx"
            quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
            line += f"   int8 {q.stat().st_size / 1e6:5.1f} MB"
        print(line, flush=True)
    print("tokenizer: CLIP BPE (vocab + merges, openai/clip-vit-base-patch32) has to be ported separately; the front end rules are scripts/text/text_frontend.py + data/text_frontend.json")


if __name__ == "__main__":
    main()
