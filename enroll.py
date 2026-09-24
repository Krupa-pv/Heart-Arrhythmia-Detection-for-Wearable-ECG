"""Step B - per patient enrollment.
(enrollment = adapting to one patient. calibration only means ECE here)

For each DS2 patient enroll on the first T sec (30, 60, 120, 300) and test on
every beat after 300 s. test beats are the same for every T so the curve is fair,
and enrollment beats never end up in the test set.

Two methods, backbone frozen in both:
  finetune - a few SGD steps on just the linear head (-> MLUpdateTask on phone)
  proto    - nearest class prototype in [embedding, RR] space, enrollment pulls
             the prototypes toward the patients mean by alpha (-> swift head)
Labels:
  realistic - every enrollment beat called N, phone has no annotations. main result
  oracle    - real MIT-BIH labels, the upper bound
  random    - random labels at 60 s, control, this should not help

recovered share = (enrolled - population) / (oracle@300 - population), per method, pooled macro-F1
alpha is picked on DS1 val patients, never DS2.

run: python enroll.py --ckpt runs/baseline/model.pt --out runs/enroll
"""
import argparse
import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from metrics import report
from model import BeatNet
from prep import DS1, DS2
from train import VAL_RECS, get_device, load, subset

ENROLL_S = [30, 60, 120, 300]
TEST_START_S = 300
ALPHAS = [0.25, 0.5, 0.75, 1.0]
PACED = [102, 104, 107, 217]
MIN_DENOM = 0.02
SIDE_REC = 232  # most of DS2's S beats, extra score without it. only for reporting, never used to pick anything


# ---------- helpers ----------
def get(r, k, sub=None):
    v = r[k] if sub is None else r[k][sub]
    return np.nan if v is None else v


@torch.no_grad()
def embed(model, d, dev, bs=4096):
    """64-d embedding + 4 RR features for each beat"""
    model.eval()
    x, rr = torch.from_numpy(d["x"]).to(dev), torch.from_numpy(d["rr"]).to(dev)
    return torch.cat([torch.cat([model.backbone(x[i:i + bs]), rr[i:i + bs]], 1)
                      for i in range(0, len(x), bs)])


def finetune(head, z, y, steps, lr):
    h = copy.deepcopy(head)
    h.train()
    opt = torch.optim.SGD(h.parameters(), lr=lr)  # plain SGD so MLUpdateTask can do the same
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(h(z), y).backward()
        opt.step()
    return h


@torch.no_grad()
def head_pred(head, z):
    head.eval()
    return head(z).argmax(1).cpu().numpy()


def proto_enroll(P_pop, z, y, alpha):
    P = P_pop.clone()
    for c in y.unique():
        P[c] = (1 - alpha) * P[c] + alpha * z[y == c].mean(0)
    return P


@torch.no_grad()
def proto_pred(P, z, sd):
    return torch.cdist(z / sd, P / sd).argmin(1).cpu().numpy()


def split_patient(d, r):
    """enrollment beat indices for each T, plus the test beats (same for all T)"""
    m = d["rec"] == r
    t0 = d["t"][m].min()
    enroll = {T: np.flatnonzero(m & (d["t"] < t0 + T)) for T in ENROLL_S}
    test = np.flatnonzero(m & (d["t"] >= t0 + TEST_START_S))
    assert d["t"][enroll[max(ENROLL_S)]].max() < d["t"][test].min(), f"leak in record {r}"
    return enroll, test


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="runs/baseline/model.pt")
    ap.add_argument("--out", default="runs/enroll")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = get_device()
    os.makedirs(a.out, exist_ok=True)

    ds1, ds2 = load(f"{a.data}/ds1.npz"), load(f"{a.data}/ds2.npz")

    # --- leakage checks ---
    r1, r2 = set(np.unique(ds1["rec"]).tolist()), set(np.unique(ds2["rec"]).tolist())
    assert not r1 & r2, "DS1/DS2 overlap"
    assert r1 <= set(DS1) and r2 <= set(DS2), "unexpected records"
    assert not (r1 | r2) & set(PACED), "paced record present"

    model = BeatNet().to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    head = model.head

    # population prototypes and scale, from DS1 train patients only
    is_val = np.isin(ds1["rec"], VAL_RECS)
    tr1 = subset(ds1, ~is_val)
    z_tr, y_tr = embed(model, tr1, dev), torch.from_numpy(tr1["y"]).to(dev)
    P_pop = torch.stack([z_tr[y_tr == c].mean(0) if (y_tr == c).any()
                         else torch.full_like(z_tr[0], 1e6) for c in range(5)])
    sd = z_tr.std(0) + 1e-6

    def run_patients(d, recs, fn):
        z, y = embed(model, d, dev), torch.from_numpy(d["y"]).to(dev)
        out = {}
        for r in recs:
            enroll, test = split_patient(d, r)
            out[r] = (fn(z, y, enroll, test), d["y"][test])
        return out

    # --- pick alpha on DS1 val patients (realistic labels, 60 s) ---
    va = subset(ds1, is_val)
    alpha_scores = {}
    for al in ALPHAS:
        res = run_patients(va, VAL_RECS, lambda z, y, e, t, al=al: proto_pred(
            proto_enroll(P_pop, z[e[60]], torch.zeros_like(y[e[60]]), al), z[t], sd))
        alpha_scores[al] = report(np.concatenate([v[1] for v in res.values()]),
                                  np.concatenate([v[0] for v in res.values()]))["macro_f1"]
    alpha = max(alpha_scores, key=alpha_scores.get)
    print("alpha on DS1 val:", {k: round(v, 3) for k, v in alpha_scores.items()}, "-> chose", alpha)

    # --- DS2 conditions ---
    recs = sorted(r2)
    z2, y2 = embed(model, ds2, dev), torch.from_numpy(ds2["y"]).to(dev)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    conds = {}

    def add(name, r, p):
        conds.setdefault(name, {})[r] = p

    y_test = {}
    for r in recs:
        enroll, test = split_patient(ds2, r)
        zt = z2[test]
        y_test[r] = ds2["y"][test]
        add("finetune/population", r, head_pred(head, zt))
        add("proto/population", r, proto_pred(P_pop, zt, sd))
        for T in ENROLL_S:
            ze, ye = z2[enroll[T]], y2[enroll[T]]
            labels = {"realistic": torch.zeros_like(ye), "oracle": ye}
            if T == 60:
                labels["random"] = torch.randint(0, 5, ye.shape, generator=g).to(dev)
            for cond, yl in labels.items():
                add(f"finetune/{cond}@{T}", r, head_pred(finetune(head, ze, yl, a.steps, a.lr), zt))
                add(f"proto/{cond}@{T}", r, proto_pred(proto_enroll(P_pop, ze, yl, alpha), zt, sd))

    # --- score, pooled over all DS2 test beats ---
    y_pool = np.concatenate([y_test[r] for r in recs])
    pooled = {k: report(y_pool, np.concatenate([v[r] for r in recs])) for k, v in conds.items()}
    per_pt = {k: {r: report(y_test[r], v[r]) for r in recs} for k, v in conds.items()}
    recs_x = [r for r in recs if r != SIDE_REC]
    y_pool_x = np.concatenate([y_test[r] for r in recs_x])
    pooled_x = {k: report(y_pool_x, np.concatenate([v[r] for r in recs_x])) for k, v in conds.items()}

    summary, warnings = {}, []
    for method in ["finetune", "proto"]:
        f_pop = get(pooled[f"{method}/population"], "macro_f1")
        f_orc = get(pooled[f"{method}/oracle@300"], "macro_f1")
        f_pop_x = get(pooled_x[f"{method}/population"], "macro_f1")
        denom = f_orc - f_pop
        if denom < MIN_DENOM:
            warnings.append(f"{method}: oracle@300 - population = {denom:+.3f} (< {MIN_DENOM}). "
                            "Recovered share is unstable; report absolute macro-F1 deltas instead.")
        for k, rep in pooled.items():
            if not k.startswith(method + "/"):
                continue
            f, rx = get(rep, "macro_f1"), pooled_x[k]
            summary[k] = {"macro_f1": f, "delta_vs_pop": f - f_pop,
                          "V_sens": get(rep, "V", "sens"), "S_sens": get(rep, "S", "sens"),
                          "recovered_share": (f - f_pop) / denom if denom >= MIN_DENOM else np.nan,
                          f"macro_f1_no{SIDE_REC}": get(rx, "macro_f1"),
                          f"delta_vs_pop_no{SIDE_REC}": get(rx, "macro_f1") - f_pop_x,
                          f"V_sens_no{SIDE_REC}": get(rx, "V", "sens"),
                          f"S_sens_no{SIDE_REC}": get(rx, "S", "sens")}

    print(f"\n{'condition':<26}{'macroF1':>9}{'Δ vs pop':>10}{'V sens':>9}{'S sens':>9}{'recovered':>11}"
          f"  | no {SIDE_REC}:{'macroF1':>8}{'Δ vs pop':>10}{'V sens':>9}{'S sens':>9}")
    x = f"_no{SIDE_REC}"
    for k, s in summary.items():
        print(f"{k:<26}{s['macro_f1']:>9.3f}{s['delta_vs_pop']:>+10.3f}{s['V_sens']:>9.3f}"
              f"{s['S_sens']:>9.3f}{s['recovered_share']:>11.1%}  |        "
              f"{s['macro_f1' + x]:>8.3f}{s['delta_vs_pop' + x]:>+10.3f}{s['V_sens' + x]:>9.3f}{s['S_sens' + x]:>9.3f}")
    for w in warnings:
        print("WARNING:", w)

    json.dump({"args": vars(a), "alpha": alpha, "alpha_val_scores": alpha_scores,
               "summary": summary, "warnings": warnings, "pooled": pooled,
               f"pooled_no{SIDE_REC}": pooled_x, "per_patient": per_pt},
              open(f"{a.out}/results.json", "w"), indent=1, default=float)
    # what the swift prototype head needs later
    json.dump({"alpha": alpha, "prototypes": P_pop.cpu().tolist(), "scale": sd.cpu().tolist(),
               "classes": ["N", "S", "V", "F", "Q"], "feature_order": "embedding[64] + rr[4]"},
              open(f"{a.out}/proto_constants.json", "w"))

    # --- plot 1: per-patient before/after at 60 s, realistic labels ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, method in zip(axes, ["finetune", "proto"]):
        x = [get(per_pt[f"{method}/population"][r], "macro_f1") for r in recs]
        y = [get(per_pt[f"{method}/realistic@60"][r], "macro_f1") for r in recs]
        ax.scatter(x, y, s=18)
        for r, xi, yi in zip(recs, x, y):
            ax.annotate(str(r), (xi, yi), fontsize=6, alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=f"{method}: 60 s enrollment, no annotations", xlabel="population macro-F1",
               xlim=(0, 1.02), ylim=(0, 1.02))
    axes[0].set_ylabel("after enrollment macro-F1 (above line = helped)")
    fig.tight_layout(); fig.savefig(f"{a.out}/per_patient_60s.png", dpi=200)

    # --- plot 2: enrollment-length curve ---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, c in [("finetune", "C0"), ("proto", "C1")]:
        for cond, ls in [("realistic", "-"), ("oracle", "--")]:
            ax.plot(ENROLL_S, [summary[f"{method}/{cond}@{T}"]["delta_vs_pop"] for T in ENROLL_S],
                    ls, marker="o", c=c, label=f"{method}, {cond} labels")
        ax.scatter([60], [summary[f"{method}/random@60"]["delta_vs_pop"]], marker="x", c=c,
                   label=f"{method}, random labels")
    ax.axhline(0, c="k", lw=1)
    ax.set(xscale="log", xticks=ENROLL_S, xticklabels=["30 s", "60 s", "2 min", "5 min"],
           xlabel="enrollment length", ylabel="macro-F1 change vs. population")
    ax.minorticks_off()  # log axis adds its own labels like 4x10^1 otherwise
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{a.out}/enrollment_curve.png", dpi=200)
    print(f"\nresults.json, proto_constants.json, plots in {a.out}/")


if __name__ == "__main__":
    main()
