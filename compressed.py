"""Does the enrollment gain survive compression?

Embeds every DS2 beat with the FP16 and 6-bit Core ML embedding models (batch 1, CPU, like
the phone ran them), then runs the prototype head with the constants the phone ships
(proto_constants.json, made from the pytorch model). pytorch row is there to check this
reproduces enroll.py.

needs a Mac
run: python compressed.py
"""
import argparse
import json
import os

import coremltools as ct
import numpy as np
import torch

from enroll import SIDE_REC, embed, get, proto_enroll, proto_pred, split_patient
from metrics import report
from model import BeatNet
from train import load

VARIANTS = {"fp16": "ecg_embedding_fp16.mlpackage", "pal6": "ecg_embedding_pal6.mlpackage"}


def coreml_embed(path, x):
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    return np.stack([m.predict({"beat": x[i:i + 1]})["embedding"][0] for i in range(len(x))])


def run_head(z, ds2, c):
    """population, realistic@60, oracle@300 predictions on each patient's test beats"""
    P = torch.tensor(c["prototypes"], dtype=torch.float64)
    sd = torch.tensor(c["scale"], dtype=torch.float64)
    z = torch.as_tensor(z, dtype=torch.float64)
    y = torch.from_numpy(ds2["y"])
    preds = {"population": {}, "realistic@60": {}, "oracle@300": {}}
    y_test = {}
    for r in sorted(np.unique(ds2["rec"]).tolist()):
        enroll, test = split_patient(ds2, r)
        y_test[r] = ds2["y"][test]
        zt = z[test]
        preds["population"][r] = proto_pred(P, zt, sd)
        e = enroll[60]
        preds["realistic@60"][r] = proto_pred(proto_enroll(P, z[e], torch.zeros(len(e), dtype=torch.long), c["alpha"]), zt, sd)
        e = enroll[300]
        preds["oracle@300"][r] = proto_pred(proto_enroll(P, z[e], y[e], c["alpha"]), zt, sd)
    return preds, y_test


def score(preds, y_test):
    out = {}
    for recs, tag in [(sorted(y_test), ""), ([r for r in sorted(y_test) if r != SIDE_REC], f"_no{SIDE_REC}")]:
        yp = np.concatenate([y_test[r] for r in recs])
        for k, v in preds.items():
            rep = report(yp, np.concatenate([v[r] for r in recs]))
            out.setdefault(k, {}).update({"macro_f1" + tag: get(rep, "macro_f1"),
                                          "V_sens" + tag: get(rep, "V", "sens"),
                                          "S_sens" + tag: get(rep, "S", "sens")})
    pop, orc = out["population"]["macro_f1"], out["oracle@300"]["macro_f1"]
    for k in out:
        out[k]["delta"] = out[k]["macro_f1"] - pop
        out[k]["recovered_share"] = (out[k]["macro_f1"] - pop) / (orc - pop)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="runs/baseline/model.pt")
    ap.add_argument("--consts", default="runs/enroll/proto_constants.json")
    ap.add_argument("--models", default="runs/coreml")
    ap.add_argument("--out", default="runs/compressed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds2 = load(f"{a.data}/ds2.npz")
    c = json.load(open(a.consts))

    model = BeatNet()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    emb = {"pytorch": embed(model, ds2, "cpu").numpy()[:, :64]}
    for name, f in VARIANTS.items():
        emb[name] = coreml_embed(os.path.join(a.models, f), ds2["x"])
        print(f"{name}: embedded {len(emb[name])} beats, max |diff| vs pytorch "
              f"{np.abs(emb[name] - emb['pytorch']).max():.3e}")

    results = {}
    for name, e in emb.items():
        preds, y_test = run_head(np.concatenate([e, ds2["rr"]], 1), ds2, c)
        results[name] = score(preds, y_test)

    print(f"\n{'variant':<9}{'condition':<14}{'macroF1':>9}{'Δ vs pop':>10}{'V sens':>8}{'S sens':>8}"
          f"{'recovered':>11}  | no {SIDE_REC}:{'macroF1':>8}")
    for name, res in results.items():
        for k, s in res.items():
            print(f"{name:<9}{k:<14}{s['macro_f1']:>9.3f}{s['delta']:>+10.3f}{s['V_sens']:>8.3f}"
                  f"{s['S_sens']:>8.3f}{s['recovered_share']:>11.1%}  |        {s[f'macro_f1_no{SIDE_REC}']:>8.3f}")
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=1, default=float)
    print(f"\nwrote {a.out}/results.json")


if __name__ == "__main__":
    main()
