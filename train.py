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


def augment(x, max_shift=8, scale=0.2, noise=0.05):
    """random shift (edge padded), amplitude scale and noise per beat, so the CNN can't just
    memorize the exact shape of the training patients' beats"""
    b, n = x.shape
    s = torch.randint(-max_shift, max_shift + 1, (b, 1), device=x.device)
    idx = (torch.arange(n, device=x.device)[None] + s).clamp(0, n - 1)
    x = x.gather(1, idx)
    x = x * (1 + scale * (2 * torch.rand(b, 1, device=x.device) - 1))
    return x + noise * torch.randn_like(x)


@torch.no_grad()
def predict(model, d, dev, bs=4096):
    model.eval()
    x, rr, _ = tensors(d, dev)
    return torch.cat([model(x[i:i + bs], rr[i:i + bs]).argmax(1)
                      for i in range(0, len(x), bs)]).cpu().numpy()


def fit(tr, dev, epochs, va=None, ckpt=None, seed=0, weights="sqrt", aug=False, dropout=0.0,
        select="val", on_epoch=None):
    """trains BeatNet. select=val keeps the best epoch by val macro-F1, select=last keeps the last
    one (val still reported). on_epoch(ep, model) gets called after every epoch, used by cv.py"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    counts = np.bincount(tr["y"], minlength=5).astype(float)
    if weights == "inv":
        w = 1 / np.maximum(counts, 1)
    else:
        w = 1 / np.sqrt(np.maximum(counts, 1))
    w = w / w.sum() * 5  # default 1/sqrt(count), plain 1/count pushes Q and F way too hard
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))

    model = BeatNet(dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, rr, y = tensors(tr, dev)

    best, best_ep, best_rep = -1.0, epochs - 1, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y), device=dev)
        for i in range(0, len(y), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            xb = augment(x[idx]) if aug else x[idx]
            loss_fn(model(xb, rr[idx]), y[idx]).backward()
            opt.step()
        if on_epoch:
            on_epoch(ep, model)
        if va is not None:
            rep = report(va["y"], predict(model, va, dev))
            f1 = rep["macro_f1"]
            print(f"epoch {ep:2d}  val macro-F1 {f1:.3f}")
            if select == "last" or f1 > best:
                best, best_ep, best_rep = f1, ep, rep
                torch.save(model.state_dict(), ckpt)
    if va is not None:
        model.load_state_dict(torch.load(ckpt))
    elif ckpt:
        torch.save(model.state_dict(), ckpt)
    return model, best_ep, best, best_rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", choices=["sqrt", "inv"], default="sqrt")
    ap.add_argument("--no-test", action="store_true", help="don't touch DS2, for picking settings")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--select", choices=["val", "last"], default="val",
                    help="val = best epoch on val patients, last = fixed --epochs (picked by cv.py)")
    a = ap.parse_args()

    dev = get_device()
    os.makedirs(a.out, exist_ok=True)
    ds1 = load(f"{a.data}/ds1.npz")
    is_val = np.isin(ds1["rec"], VAL_RECS)

    model, best_ep, best, val_rep = fit(subset(ds1, ~is_val), dev, a.epochs, va=subset(ds1, is_val),
                                        ckpt=f"{a.out}/model.pt", seed=a.seed, weights=a.weights,
                                        aug=a.aug, dropout=a.dropout, select=a.select)
    json.dump({"best_epoch": best_ep, "val": val_rep, "args": vars(a)},
              open(f"{a.out}/val_results.json", "w"), indent=1)
    if a.no_test:
        print(f"\nbest epoch {best_ep}  (val macro-F1 {best:.3f})\nDS1 val:\n" + fmt(val_rep))
        return
    ds2 = load(f"{a.data}/ds2.npz")

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
