"""Patient-grouped cross validation on DS1, for picking training settings without DS2.

5 folds, split by patient (S and V heavy patients spread across folds). For every epoch the
held-out patients' predictions get pooled over all folds, so each setting is scored on all 22
DS1 patients instead of 4. Epoch count is picked on a 5-epoch moving average of pooled macro-F1
so one lucky epoch doesn't win.

run: python cv.py --name aug --aug     (same training flags as train.py)
"""
import argparse
import json
import os

import numpy as np

from metrics import report
from train import fit, get_device, load, predict, subset

SMOOTH = 5


def make_folds(d, k):
    """patients sorted by S count (then V), dealt out snake order 0..k-1, k-1..0, so every fold
    gets 4-5 patients and the S heavy ones end up in different folds"""
    recs = np.unique(d["rec"])
    cnt = {r: ((d["y"][d["rec"] == r] == 1).sum(), (d["y"][d["rec"] == r] == 2).sum()) for r in recs}
    folds = [[] for _ in range(k)]
    for i, r in enumerate(sorted(recs, key=lambda r: (-cnt[r][0], -cnt[r][1]))):
        f = i % k if (i // k) % 2 == 0 else k - 1 - i % k
        folds[f].append(int(r))
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/cv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", choices=["sqrt", "inv"], default="sqrt")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    a = ap.parse_args()

    dev = get_device()
    ds1 = load(f"{a.data}/ds1.npz")  # DS2 is never loaded here
    folds = make_folds(ds1, a.folds)
    print("folds:", folds)

    preds = np.zeros((a.epochs, len(ds1["y"])), np.int64)
    for f, recs in enumerate(folds):
        held = np.isin(ds1["rec"], recs)
        va = subset(ds1, held)

        def on_epoch(ep, model, held=held, va=va):
            preds[ep, held] = predict(model, va, dev)

        fit(subset(ds1, ~held), dev, a.epochs, seed=a.seed, weights=a.weights, aug=a.aug,
            dropout=a.dropout, on_epoch=on_epoch)
        print(f"fold {f} done ({recs})")

    reps = [report(ds1["y"], preds[ep]) for ep in range(a.epochs)]
    f1 = np.array([r["macro_f1"] for r in reps])
    w = min(SMOOTH, a.epochs)
    smooth = np.convolve(f1, np.ones(w) / w, mode="valid")  # smooth[i] = mean f1[i:i+w]
    ep_pick = int(np.argmax(smooth)) + w // 2  # center of the best window
    r = reps[ep_pick]
    out = {"args": vars(a), "folds": folds, "f1_per_epoch": f1.tolist(), "epochs_picked": ep_pick + 1,
           "smoothed_f1": float(smooth.max()), "at_pick": r}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/{a.name}.json", "w"), indent=1, default=float)
    print(f"\n{a.name}: smoothed pooled macro-F1 {smooth.max():.3f}, train for {ep_pick + 1} epochs"
          f" | at that epoch: F1 {r['macro_f1']:.3f}  S sens {r['S']['sens']:.3f} ppv {r['S']['ppv'] or 0:.3f}"
          f"  V sens {r['V']['sens']:.3f} ppv {r['V']['ppv'] or 0:.3f}")


if __name__ == "__main__":
    main()
