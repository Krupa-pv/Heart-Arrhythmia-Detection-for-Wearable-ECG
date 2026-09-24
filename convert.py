"""Step 7 - PyTorch to Core ML (ML Program).

First checks FP32 Core ML matches PyTorch on all of DS2, dont trust anything before that.
Then exports just the embedding model (backbone) as FP16 and 6-bit palettized, batch 1,
for the swift prototype head. swift adds the 4 RR values, divides by scale and uses
prototypes/alpha from runs/enroll/proto_constants.json.

needs a Mac (Core ML predict only works on macOS)
run: python convert.py --ckpt runs/baseline/model.pt --out runs/coreml
"""
import argparse
import json
import os

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch

from model import BeatNet
from train import load

B = 1024


def convert(module, example, inputs, out_name, precision):
    traced = torch.jit.trace(module.eval(), example)
    return ct.convert(traced, convert_to="mlprogram", compute_precision=precision,
                      inputs=inputs, outputs=[ct.TensorType(name=out_name)],
                      minimum_deployment_target=ct.target.iOS17)


def dir_size_kb(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="runs/baseline/model.pt")
    ap.add_argument("--out", default="runs/coreml")
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = BeatNet()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model.eval()
    ds2 = load(f"{a.data}/ds2.npz")

    # ---- parity check, FP32 core ml vs pytorch on every DS2 beat ----
    full = convert(model, (torch.randn(B, 256), torch.randn(B, 4)),
                   [ct.TensorType(name="beat", shape=(B, 256), dtype=np.float32),
                    ct.TensorType(name="rr", shape=(B, 4), dtype=np.float32)],
                   "logits", ct.precision.FLOAT32)
    n, cm_logits = len(ds2["y"]), []
    for i in range(0, n, B):
        xb, rb = ds2["x"][i:i + B], ds2["rr"][i:i + B]
        k = len(xb)
        if k < B:
            xb = np.concatenate([xb, np.zeros((B - k, 256), np.float32)])
            rb = np.concatenate([rb, np.zeros((B - k, 4), np.float32)])
        cm_logits.append(full.predict({"beat": xb, "rr": rb})["logits"][:k])
    cm_logits = np.concatenate(cm_logits)
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(ds2["x"]), torch.from_numpy(ds2["rr"])).numpy()
    diff = float(np.abs(cm_logits - pt_logits).max())
    agree = float((cm_logits.argmax(1) == pt_logits.argmax(1)).mean())
    ok = diff < a.tol
    print(f"parity FP32: max |Δlogit| = {diff:.2e}, argmax agreement = {agree:.5f}  -> {'PASS' if ok else 'FAIL'}")

    # ---- embedding models for the phone, FP16 and 6-bit ----
    emb_in = [ct.TensorType(name="beat", shape=(1, 256), dtype=np.float32)]
    fp16 = convert(model.backbone, (torch.randn(1, 256),), emb_in, "embedding", ct.precision.FLOAT16)
    pal6 = cto.palettize_weights(fp16, config=cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=6, weight_threshold=0)))
    sizes = {}
    for name, m in [("fp16", fp16), ("pal6", pal6)]:
        path = f"{a.out}/ecg_embedding_{name}.mlpackage"
        m.save(path)
        sizes[name] = round(dir_size_kb(path), 1)
        print(f"saved {path}  ({sizes[name]} KB)")

    json.dump({"parity_fp32": {"max_abs_logit_diff": diff, "argmax_agreement": agree, "pass": ok},
               "embedding_model_size_kb": sizes},
              open(f"{a.out}/coreml_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
