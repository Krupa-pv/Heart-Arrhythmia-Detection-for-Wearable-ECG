"""Replay a DS2 record through the swift prototype head and check it matches python.

export: dump one DS2 record (beats, RR, labels, times) to json for the swift replay tool
check:  compare swift output with python
  1. swift embeddings vs coremltools embeddings (same .mlpackage)
  2. python prototype head run on swift's embeddings -> must give the same predictions
  3. swift predictions vs the pytorch path in enroll.py (proto/realistic@60)

run: python replay.py export --rec 214
     swift run -c release --package-path swift/ProtoHead replay runs/replay/214.json ...
     python replay.py check --rec 214

finetune: same idea for the MLUpdateTask head (swift finetune tool writes <rec>_finetune.json)
  1. updated fc weights, core ml vs pytorch finetune() on the same features
  2. predictions after the update, same features
  3. vs the pytorch-embedding path in enroll.py (finetune/realistic@60, finetune/oracle@300)
"""
import argparse
import json
import os

import numpy as np
import torch

from enroll import embed, finetune as pt_finetune, head_pred, proto_enroll, proto_pred, split_patient
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
    dn = load("data_norm/ds2.npz")  # v3 timing model features, same beat order
    rr_norm = dn["rr"][ds2["rec"] == a.rec]
    assert (dn["t"][ds2["rec"] == a.rec] == d["t"]).all()
    json.dump({"rec": a.rec, "t": d["t"].tolist(), "y": d["y"].tolist(), "rr": d["rr"].tolist(),
               "rr_norm": rr_norm.tolist(), "x": d["x"].tolist()}, open(path, "w"))
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


def check_finetune(a):
    ds2 = load(f"{a.data}/ds2.npz")
    d = subset(ds2, ds2["rec"] == a.rec)
    sw = json.load(open(f"{a.out}/{a.rec}_finetune.json"))
    enroll_all, test = split_patient(ds2, a.rec)
    off = np.flatnonzero(ds2["rec"] == a.rec)[0]
    test = test - off
    assert sw["test_idx"] == test.tolist(), "swift picked different test beats"
    y = d["y"][test]

    model = BeatNet()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    z_sw = torch.from_numpy(np.concatenate([np.array(sw["embedding"], np.float32), d["rr"]], 1))
    z_pt = embed(model, d, "cpu")
    yt = torch.from_numpy(d["y"])
    res = json.load(open(a.results))["per_patient"]

    p_pop = head_pred(model.head, z_sw[test])
    print(f"population  core ml vs pytorch head, same features: {(np.array(sw['population']) == p_pop).mean():.5f} agree")
    for name, T, oracle in [("realistic@60", 60, False), ("oracle@300", 300, True)]:
        e = enroll_all[T] - off
        labels = yt[e] if oracle else torch.zeros(len(e), dtype=torch.long)
        h = pt_finetune(model.head, z_sw[e], labels, 30, 0.01)
        w_pt = h.fc.weight.detach().numpy().ravel()
        w_cm = np.array(sw["conds"][name]["weights"], np.float32)
        p_cm = np.array(sw["conds"][name]["pred"])
        p_same = head_pred(h, z_sw[test])
        h2 = pt_finetune(model.head, z_pt[e], labels, 30, 0.01)
        p_pt = head_pred(h2, z_pt[test])
        ref = res[f"finetune/{name}"][str(a.rec)]["macro_f1"]
        print(f"{name}: update took {sw['conds'][name]['seconds'] * 1000:.1f} ms on {len(e)} beats")
        print(f"  1. weights    core ml vs pytorch, same features: max |diff| {np.abs(w_cm - w_pt).max():.2e}"
              f"  (weights moved by {np.abs(w_pt - model.head.fc.weight.detach().numpy().ravel()).max():.2e})")
        print(f"  2. preds      same features: {(p_cm == p_same).mean():.5f} agree ({(p_cm != p_same).sum()} differ)")
        print(f"  3. end to end vs pytorch embeddings: {(p_cm == p_pt).mean():.5f} agree ({(p_cm != p_pt).sum()} differ)")
        print(f"     macro-F1  core ml {report(y, p_cm)['macro_f1']:.4f}   pytorch {report(y, p_pt)['macro_f1']:.4f}"
              f"   enroll.py {ref:.4f}")


def check_v3(a):
    """swift v3 vs python fusion.py on the same record"""
    from fusion import fuse, log_probs, rr_models  # noqa: F401  (fuse is the reference)
    from train import VAL_RECS
    ds2, dn = load(f"{a.data}/ds2.npz"), load("data_norm/ds2.npz")
    m = ds2["rec"] == a.rec
    d = subset(ds2, m)
    X2 = dn["rr"][m]
    sw = json.load(open(f"{a.out}/{a.rec}_v3.json"))
    p_sw = np.array(sw["pred"])

    # rebuild the exact timing model fusion.py used, and the CNN
    ds1, ds1n = load(f"{a.data}/ds1.npz"), load("data_norm/ds1.npz")
    tr = ~np.isin(ds1["rec"], VAL_RECS)
    sc, binary, multi = rr_models(ds1n["rr"][tr], ds1["y"][tr])
    w = json.load(open(a.v3_consts))["w"]
    model = BeatNet()
    model.load_state_dict(torch.load(a.v3_ckpt, map_location="cpu"))

    # 1. same embeddings (swift's), python head + timing model
    e_sw = torch.from_numpy(np.array(sw["embedding"], np.float32))
    with torch.no_grad():
        lp_same = torch.log_softmax(model.head(torch.cat([e_sw, torch.from_numpy(d["rr"])], 1)), 1).numpy()
    p_same = fuse(lp_same, sc, binary, multi, X2, "multi", w)
    print(f"1. head + timing model, swift vs python on swift embeddings: {(p_sw == p_same).mean():.5f} agree "
          f"({(p_sw != p_same).sum()} differ)")
    # 2. end to end vs pytorch
    p_pt = fuse(log_probs(model, d, "cpu"), sc, binary, multi, X2, "multi", w)
    ref = json.load(open(a.v3_results))["per_ckpt"][a.v3_ckpt]["per_patient"][str(a.rec)]["macro_f1"]
    print(f"2. end to end, swift (fp16 core ml) vs pytorch: {(p_sw == p_pt).mean():.5f} agree "
          f"({(p_sw != p_pt).sum()} of {len(p_sw)} differ)")
    print(f"   macro-F1  swift {report(d['y'], p_sw)['macro_f1']:.4f}   python {report(d['y'], p_pt)['macro_f1']:.4f}"
          f"   fusion.py ds2.json {ref:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "check", "finetune", "v3"])
    ap.add_argument("--rec", type=int, default=214)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/replay")
    ap.add_argument("--consts", default="runs/enroll/proto_constants.json")
    ap.add_argument("--results", default="runs/enroll/results.json")
    ap.add_argument("--ckpt", default="runs/baseline/model.pt")
    ap.add_argument("--mlpackage", default="runs/coreml/ecg_embedding_fp16.mlpackage")
    ap.add_argument("--v3-ckpt", default="runs/v3_baseline/model.pt")
    ap.add_argument("--v3-consts", default="runs/fusion/v3_constants.json")
    ap.add_argument("--v3-results", default="runs/fusion/ds2.json")
    a = ap.parse_args()
    {"export": export, "check": check, "finetune": check_finetune, "v3": check_v3}[a.cmd](a)


if __name__ == "__main__":
    main()
