import numpy as np

CLASSES = ["N", "S", "V", "F", "Q"]
F1_CLASSES = [0, 1, 2, 3]  # Q is tiny (7 beats in DS2) so its left out of macro-F1


def confusion(y, p, k=5):
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, p), 1)
    return cm


def report(y, p):
    cm = confusion(y, p)
    out = {}
    f1s = []
    for c in range(5):
        tp, sup, pred = cm[c, c], cm[c].sum(), cm[:, c].sum()
        sens = tp / sup if sup else None
        ppv = tp / pred if pred else None
        f1 = 2 * sens * ppv / (sens + ppv) if sens and ppv else (0.0 if sup else None)
        out[CLASSES[c]] = {"sens": sens, "ppv": ppv, "f1": f1, "n": int(sup)}
        if c in F1_CLASSES and sup:
            f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else None
    out["confusion"] = cm.tolist()
    return out


def fmt(r):
    s = lambda v: "  -  " if v is None else f"{v:.3f}"
    lines = [f"macro-F1 (N,S,V,F present): {s(r['macro_f1'])}"]
    for c in CLASSES:
        d = r[c]
        lines.append(f"  {c}: sens {s(d['sens'])}  ppv {s(d['ppv'])}  n={d['n']}")
    return "\n".join(lines)


def ece(probs, y, bins=15):
    """expected calibration error, how far confidence is from accuracy (binned)"""
    conf, acc = probs.max(1), (probs.argmax(1) == y)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)
