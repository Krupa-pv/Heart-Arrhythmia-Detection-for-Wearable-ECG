"""Replay a DS2 record through the swift prototype head and check it matches python.

export: dump one DS2 record (beats, RR, labels, times) to json for the swift replay tool
check:  compare swift output with python
  1. swift embeddings vs coremltools embeddings (same .mlpackage)
  2. python prototype head run on swift's embeddings -> must give the same predictions
  3. swift predictions vs the pytorch path in enroll.py (proto/realistic@60)

run: python replay.py export --rec 214
     swift run -c release --package-path swift/ProtoHead replay runs/replay/214.json ...
     python replay.py check --rec 214
"""
import argparse
import json
import os

import numpy as np
import torch

from enroll import embed, proto_enroll, proto_pred, split_patient
from metrics import report
from model import BeatNet
from train import load, subset

ENROLL_T = 60


def export(a):
    ds2 = load(f"{a.data}/ds2.npz")
    d = subset(ds2, ds2["rec"] == a.rec)
    assert len(d["y"]), f"record {a.rec} not in DS2"
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/{a.rec}.json"
    json.dump({"rec": a.rec, "t": d["t"].tolist(), "y": d["y"].tolist(),
               "rr": d["rr"].tolist(), "x": d["x"].tolist()}, open(path, "w"))
    print(f"wrote {path}  ({len(d['y'])} beats)")


def head_preds(z, c, enroll, test):
    """python prototype head, same steps as enroll.py realistic labels"""
    P = torch.tensor(c["prototypes"], dtype=torch.float64)
    sd = torch.tensor(c["scale"], dtype=torch.float64)
    z = torch.as_tensor(z, dtype=torch.float64)
    ze = z[enroll]
    Pe = proto_enroll(P, ze, torch.zeros(len(ze), dtype=torch.long), c["alpha"])
    return proto_pred(Pe, z[test], sd)


def check(a):
    ds2 = load(f"{a.data}/ds2.npz")
    d = subset(ds2, ds2["rec"] == a.rec)
    sw = json.load(open(f"{a.out}/{a.rec}_swift.json"))
    c = json.load(open(a.consts))
    enroll_all, test = split_patient(ds2, a.rec)
    off = np.flatnonzero(ds2["rec"] == a.rec)[0]  # record indices -> local indices
    enroll, test = enroll_all[ENROLL_T] - off, test - off
    y = d["y"][test]

    assert sw["test_idx"] == test.tolist(), "swift picked different test beats"
    assert sw["enroll_idx"] == enroll.tolist(), "swift picked different enrollment beats"
    p_sw = np.array(sw["pred"])

    # 1. swift embeddings vs coremltools on the same model
    import coremltools as ct
    m = ct.models.MLModel(a.mlpackage, compute_units=ct.ComputeUnit.CPU_ONLY)
    e_ct = np.stack([m.predict({"beat": d["x"][i:i + 1]})["embedding"][0] for i in range(len(d["y"]))])
    e_sw = np.array(sw["embedding"], np.float32)
    print(f"1. embeddings  swift vs coremltools: max |diff| {np.abs(e_sw - e_ct).max():.2e}")

    # 2. same embeddings, python head -> should be identical to swift
    z_sw = np.concatenate([e_sw, d["rr"]], 1)
    p_py_same = head_preds(z_sw, c, enroll, test)
    print(f"2. head logic  swift vs python on swift embeddings: "
          f"{(p_sw == p_py_same).mean():.5f} agree ({(p_sw != p_py_same).sum()} differ)")

    # 3. vs the pytorch path used in enroll.py
    dev = "cpu"
    model = BeatNet()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    z_pt = embed(model, d, dev).numpy()
    p_pt = head_preds(z_pt, c, enroll, test)
    f_sw, f_pt = report(y, p_sw)["macro_f1"], report(y, p_pt)["macro_f1"]
    ref = json.load(open(a.results))["per_patient"]["proto/realistic@60"][str(a.rec)]["macro_f1"]
    print(f"3. end to end  swift (fp16 core ml) vs pytorch: {(p_sw == p_pt).mean():.5f} agree "
          f"({(p_sw != p_pt).sum()} of {len(p_sw)} differ)")
    print(f"   macro-F1  swift {f_sw:.4f}   python {f_pt:.4f}   enroll.py results.json {ref:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "check"])
    ap.add_argument("--rec", type=int, default=214)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/replay")
    ap.add_argument("--consts", default="runs/enroll/proto_constants.json")
    ap.add_argument("--results", default="runs/enroll/results.json")
    ap.add_argument("--ckpt", default="runs/baseline/model.pt")
    ap.add_argument("--mlpackage", default="runs/coreml/ecg_embedding_fp16.mlpackage")
    a = ap.parse_args()
    export(a) if a.cmd == "export" else check(a)


if __name__ == "__main__":
    main()
