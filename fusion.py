"""Late fusion: CNN + a linear model on RR (timing) features only.

The CNN memorizes the training patients' beat shapes and ends up ignoring timing, but timing is
what carries over to new patients for S beats (de Chazal 2004 got ~76% S with a linear model on
mostly RR features). So add a separate RR-only logistic regression and combine the two:
  s_only: CNN logits, S score += w * log-odds from a binary S-vs-rest RR model
  multi:  CNN log-probs + w * log-probs from a 4 class (N,S,V,F) RR model (Q left to the CNN)
Features: raw (the 4 in data/), norm (5 min normalized, data_norm/), or both (8).

Everything is picked by the same patient-grouped 5-fold CV on DS1 as cv.py (DS2 not loaded),
pooled over folds, averaged over CNN seeds.

run: python fusion.py --seeds 0 1 2
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cv import make_folds
from metrics import report
from train import fit, get_device, load, subset

WS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 10, 100]  # 100 ~ RR model alone
FEATS = ["raw", "norm", "both"]


@torch.no_grad()
def log_probs(model, d, dev, bs=4096):
    model.eval()
    x, rr = torch.from_numpy(d["x"]).to(dev), torch.from_numpy(d["rr"]).to(dev)
    return torch.cat([torch.log_softmax(model(x[i:i + bs], rr[i:i + bs]), 1)
                      for i in range(0, len(x), bs)]).cpu().numpy()


def rr_feats(raw, norm, kind):
    return {"raw": raw, "norm": norm, "both": np.concatenate([raw, norm], 1)}[kind]


def rr_models(X, y):
    """fit the two RR-only models on train beats: binary S vs rest, and 4 class N S V F"""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    binary = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xs, (y == 1).astype(int))
    keep = y < 4
    multi = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xs[keep], y[keep])
    return sc, binary, multi


def fuse(lp_cnn, sc, binary, multi, X, mode, w):
    s = lp_cnn.copy()
    Xs = sc.transform(X)
    if mode == "s_only":
        p = np.clip(binary.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
        s[:, 1] += w * np.log(p / (1 - p))
    else:
        lp = np.log(np.clip(multi.predict_proba(Xs), 1e-9, 1))  # classes 0..3
        s[:, :4] += w * lp
        s[:, 4] += w * lp.mean(1)  # Q: neutral, CNN decides
    return s.argmax(1)


def test(a):
    """final check on DS2, settings fixed by the CV run: timing model fit on DS1 train patients
    (same ones the CNN saw), fused with the given CNN checkpoints, scored once on DS2"""
    from model import BeatNet
    from train import VAL_RECS
    kind, mode, w = a.pick.split("/")
    w = float(w.split("=")[1])
    ds1, ds1n = load("data/ds1.npz"), load("data_norm/ds1.npz")
    ds2, ds2n = load("data/ds2.npz"), load("data_norm/ds2.npz")
    assert (ds2["y"] == ds2n["y"]).all() and (ds2["t"] == ds2n["t"]).all()
    tr = ~np.isin(ds1["rec"], VAL_RECS)
    X1, X2 = rr_feats(ds1["rr"], ds1n["rr"], kind), rr_feats(ds2["rr"], ds2n["rr"], kind)
    sc, binary, multi = rr_models(X1[tr], ds1["y"][tr])
    out = {"pick": a.pick, "per_ckpt": {}}
    for ck in a.ckpts:
        m = BeatNet()
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        lp = log_probs(m, ds2, "cpu")
        pred = fuse(lp, sc, binary, multi, X2, mode, w)
        cnn_only = report(ds2["y"], lp.argmax(1))
        fused = report(ds2["y"], pred)
        per_pt = {int(r): report(ds2["y"][ds2["rec"] == r], pred[ds2["rec"] == r]) for r in np.unique(ds2["rec"])}
        out["per_ckpt"][ck] = {"cnn_only": cnn_only, "fused": fused, "per_patient": per_pt}
        print(f"{ck}\n  CNN alone: macro-F1 {cnn_only['macro_f1']:.3f}  S sens {cnn_only['S']['sens']:.3f}"
              f"  V sens {cnn_only['V']['sens']:.3f}\n  fused:     macro-F1 {fused['macro_f1']:.3f}  "
              + "  ".join(f"{c} sens {fused[c]['sens']:.3f} ppv {fused[c]['ppv'] or 0:.3f}" for c in "NSVF"))
    # what swift needs to run the timing model: scaler + 4 class logistic regression
    out["rr_model"] = {"features": kind, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                       "coef": multi.coef_.tolist(), "intercept": multi.intercept_.tolist(), "w": w}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/ds2.json", "w"), indent=1, default=float)


def export(a):
    """everything swift needs for v3 besides the Core ML embedding model"""
    from model import BeatNet
    m = BeatNet()
    m.load_state_dict(torch.load(a.ckpts[0], map_location="cpu"))
    rr = json.load(open(f"{a.out}/ds2.json"))["rr_model"]
    assert rr["features"] == "norm", "swift side expects the 4 normalized RR features"
    c = {"head_weight": m.head.fc.weight.detach().tolist(), "head_bias": m.head.fc.bias.detach().tolist(),
         "rr_mean": rr["mean"], "rr_scale": rr["scale"], "rr_coef": rr["coef"],
         "rr_intercept": rr["intercept"], "w": rr["w"], "classes": ["N", "S", "V", "F", "Q"]}
    json.dump(c, open(f"{a.out}/v3_constants.json", "w"))
    print(f"wrote {a.out}/v3_constants.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=4)  # v3, picked by cv.py
    ap.add_argument("--out", default="runs/fusion")
    ap.add_argument("--test", action="store_true", help="final DS2 check with --pick and --ckpts")
    ap.add_argument("--export", action="store_true",
                    help="write v3_constants.json for swift: CNN head weights (first --ckpts) + timing model")
    ap.add_argument("--pick", default="norm/multi/w=3")
    ap.add_argument("--ckpts", nargs="+", default=["runs/v3_baseline/model.pt"])
    a = ap.parse_args()
    if a.test:
        return test(a)
    if a.export:
        return export(a)

    dev = get_device()
    ds1, ds1n = load("data/ds1.npz"), load("data_norm/ds1.npz")  # DS2 never loaded
    assert (ds1["y"] == ds1n["y"]).all() and (ds1["t"] == ds1n["t"]).all(), "beat order differs"
    folds = make_folds(ds1, 5)

    # out-of-fold CNN log-probs, per seed
    lp = {s: np.zeros((len(ds1["y"]), 5), np.float32) for s in a.seeds}
    for s in a.seeds:
        for f, recs in enumerate(folds):
            held = np.isin(ds1["rec"], recs)
            model, *_ = fit(subset(ds1, ~held), dev, a.epochs, seed=s)
            lp[s][held] = log_probs(model, subset(ds1, held), dev)
        print(f"cnn seed {s} done")

    # out-of-fold RR models (deterministic, same for every seed)
    results = {}
    for kind in FEATS:
        X = rr_feats(ds1["rr"], ds1n["rr"], kind)
        fitted = [(np.isin(ds1["rec"], recs), rr_models(X[~np.isin(ds1["rec"], recs)],
                                                        ds1["y"][~np.isin(ds1["rec"], recs)]))
                  for recs in folds]
        for mode in ["s_only", "multi"]:
            for w in WS:
                reps = []
                for s in a.seeds:
                    pred = np.zeros(len(ds1["y"]), np.int64)
                    for held, (sc, binary, multi) in fitted:
                        pred[held] = fuse(lp[s][held], sc, binary, multi, X[held], mode, w)
                    reps.append(report(ds1["y"], pred))
                key = f"{kind}/{mode}/w={w}"
                results[key] = {k: float(np.mean([r[c][m] or 0 for r in reps])) if c != "macro_f1" else None
                                for k, (c, m) in {"S_sens": ("S", "sens"), "S_ppv": ("S", "ppv"),
                                                   "V_sens": ("V", "sens"), "V_ppv": ("V", "ppv"),
                                                   "S_f1": ("S", "f1"), "V_f1": ("V", "f1")}.items()}
                results[key]["macro_f1"] = float(np.mean([r["macro_f1"] for r in reps]))
                results[key]["macro_f1_sd"] = float(np.std([r["macro_f1"] for r in reps]))

    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "folds": folds, "results": results}, open(f"{a.out}/cv.json", "w"), indent=1)
    base = results["raw/s_only/w=0"]
    print(f"\nCNN alone (w=0): macro-F1 {base['macro_f1']:.3f}  S sens {base['S_sens']:.3f} ppv {base['S_ppv']:.3f}"
          f"  V sens {base['V_sens']:.3f}")
    print(f"{'setting':<24}{'macroF1':>9}{'sd':>7}{'S sens':>8}{'S ppv':>7}{'S F1':>7}{'V sens':>8}{'V F1':>7}")
    for k, r in sorted(results.items(), key=lambda kv: -kv[1]["macro_f1"])[:12]:
        print(f"{k:<24}{r['macro_f1']:>9.3f}{r['macro_f1_sd']:>7.3f}{r['S_sens']:>8.3f}{r['S_ppv']:>7.3f}"
              f"{r['S_f1']:>7.3f}{r['V_sens']:>8.3f}{r['V_f1']:>7.3f}")


if __name__ == "__main__":
    main()
