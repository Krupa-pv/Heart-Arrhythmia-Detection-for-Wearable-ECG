"""Turns MIT-BIH into beat windows + RR features and saves DS1 / DS2.
Uses the de Chazal inter-patient split so no patient is in both sets.

run: python prep.py --data mitdb --out data
first run downloads the database (about 100 MB)
"""
import argparse
import os

import numpy as np
import wfdb
from scipy.ndimage import median_filter

# split from de Chazal et al. 2004. paced records 102, 104, 107, 217 are left out (AAMI)
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

CLASSES = ["N", "S", "V", "F", "Q"]
AAMI = {**{s: 0 for s in "NLRej"},
        **{s: 1 for s in "aAJS"},
        **{s: 2 for s in "VE"},
        "F": 3,
        **{s: 4 for s in "/fQ"}}

FS = 360
PRE, POST = 100, 156   # 256 samples per beat, R peak sits at index 100
RR_WIN = 10            # local RR = mean of the last 10 beats only
LONG_S = 300           # for --rr-norm long: patient's average RR over the last 5 min


def remove_baseline(sig):
    # 200 ms then 600 ms median filter gives the baseline wander, subtract it
    b = median_filter(sig, size=int(0.2 * FS) | 1)
    b = median_filter(b, size=int(0.6 * FS) | 1)
    return sig - b


def process_record(rec_id, data_dir, rr_norm="none"):
    path = os.path.join(data_dir, str(rec_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    sig = remove_baseline(rec.p_signal[:, rec.sig_name.index("MLII")])  # pick by name, 114 has MLII on channel 2

    beats = [(s, AAMI[a]) for s, a in zip(ann.sample, ann.symbol) if a in AAMI]
    r = np.array([b[0] for b in beats])
    lab = np.array([b[1] for b in beats])
    rr = np.diff(r) / FS  # in seconds, rr[i] is gap between beat i and i+1

    X, F, Y, T = [], [], [], []
    for i in range(1, len(r) - 1):
        lo, hi = r[i] - PRE, r[i] + POST
        if lo < 0 or hi > len(sig):
            continue
        w = sig[lo:hi]
        w = (w - w.mean()) / (w.std() + 1e-6)
        pre, post = rr[i - 1], rr[i]
        local = rr[max(0, i - RR_WIN):i].mean()  # only past beats so it works live too
        X.append(w)
        if rr_norm == "long":
            # divide by this patient's own average RR (past 5 min only), so a fast heart and a
            # slow heart give the same numbers for a normal beat
            j = np.searchsorted(r, r[i] - LONG_S * FS)
            avg = rr[max(0, j - 1):i].mean() if i > j else rr[:i].mean()
            F.append([pre / avg, post / avg, local / avg, pre / local])
        else:
            F.append([pre, post, local, pre / local])
        Y.append(lab[i])
        T.append(r[i] / FS)
    n = len(Y)
    return (np.array(X, np.float32), np.array(F, np.float32),
            np.array(Y, np.int64), np.full(n, rec_id, np.int32), np.array(T, np.float32))


def build(records, data_dir, rr_norm="none"):
    parts = [process_record(r, data_dir, rr_norm) for r in records]
    return {k: np.concatenate([p[j] for p in parts])
            for j, k in enumerate(["x", "rr", "y", "rec", "t"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mitdb")
    ap.add_argument("--out", default="data")
    ap.add_argument("--rr-norm", choices=["none", "long"], default="none")
    a = ap.parse_args()

    if not all(os.path.exists(os.path.join(a.data, f"{r}.dat")) for r in DS1 + DS2):
        wfdb.dl_database("mitdb", dl_dir=a.data)
    os.makedirs(a.out, exist_ok=True)

    for name, recs in [("ds1", DS1), ("ds2", DS2)]:
        d = build(recs, a.data, a.rr_norm)
        np.savez_compressed(os.path.join(a.out, f"{name}.npz"), **d)
        counts = {c: int((d["y"] == k).sum()) for k, c in enumerate(CLASSES)}
        print(f"{name}: {len(d['y'])} beats  {counts}")


if __name__ == "__main__":
    main()
