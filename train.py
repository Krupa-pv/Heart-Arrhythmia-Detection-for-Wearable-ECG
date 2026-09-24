"""Step A - the population model.
Train on DS1, pick the best epoch on a few held out DS1 patients, then
report on DS2 (overall + per patient). DS2 is not used at all in training.

run: python train.py --data data --out runs/baseline
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from metrics import fmt, report
from model import BeatNet

VAL_RECS = [106, 118, 124, 223]  # DS1 patients kept out for picking the epoch, they have V and S beats


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


def concat(*ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def tensors(d, dev):
    return (torch.from_numpy(d["x"]).to(dev), torch.from_numpy(d["rr"]).to(dev),
            torch.from_numpy(d["y"]).to(dev))


@torch.no_grad()
def predict(model, d, dev, bs=4096):
    model.eval()
    x, rr, _ = tensors(d, dev)
    return torch.cat([model(x[i:i + bs], rr[i:i + bs]).argmax(1)
                      for i in range(0, len(x), bs)]).cpu().numpy()


def fit(tr, dev, epochs, va=None, ckpt=None, seed=0):
    """trains BeatNet. if va is given keeps best epoch by val macro-F1, otherwise the last one"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    counts = np.bincount(tr["y"], minlength=5).astype(float)
    w = 1 / np.sqrt(np.maximum(counts, 1))
    w = w / w.sum() * 5  # 1/sqrt(count), plain 1/count pushes Q and F way too hard
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))

    model = BeatNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, rr, y = tensors(tr, dev)

    best, best_ep = -1.0, epochs - 1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y), device=dev)
        for i in range(0, len(y), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(model(x[idx], rr[idx]), y[idx]).backward()
            opt.step()
        if va is not None:
            f1 = report(va["y"], predict(model, va, dev))["macro_f1"]
            print(f"epoch {ep:2d}  val macro-F1 {f1:.3f}")
            if f1 > best:
                best, best_ep = f1, ep
                torch.save(model.state_dict(), ckpt)
    if va is not None:
        model.load_state_dict(torch.load(ckpt))
    elif ckpt:
        torch.save(model.state_dict(), ckpt)
    return model, best_ep, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = get_device()
    os.makedirs(a.out, exist_ok=True)
    ds1, ds2 = load(f"{a.data}/ds1.npz"), load(f"{a.data}/ds2.npz")
    is_val = np.isin(ds1["rec"], VAL_RECS)

    model, best_ep, best = fit(subset(ds1, ~is_val), dev, a.epochs,
                               va=subset(ds1, is_val), ckpt=f"{a.out}/model.pt", seed=a.seed)

    p = predict(model, ds2, dev)
    overall = report(ds2["y"], p)
    per_patient = {int(r): report(ds2["y"][ds2["rec"] == r], p[ds2["rec"] == r])
                   for r in np.unique(ds2["rec"])}

    print(f"\nbest epoch {best_ep}  (val macro-F1 {best:.3f})")
    print("DS2 overall:\n" + fmt(overall))
    json.dump({"best_epoch": best_ep, "overall": overall, "per_patient": per_patient},
              open(f"{a.out}/ds2_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
