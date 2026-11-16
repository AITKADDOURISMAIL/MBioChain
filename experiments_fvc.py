"""
Fingerprint-only experiments on FVC2002 set B (DB1-DB4, 10 fingers x 8 impressions each).

  python3 experiments_fvc.py --fvc /path/to/folder_with_DBx_B_dirs --out results_fvc

Protocol
  * enrollment : impression 1 (or majority vote of impressions 1-3 aligned on impression 1 with --multi)
  * genuine    : remaining impressions of the same finger
  * impostor   : impression 1 of every other finger of the SAME database (FVC-style), correct PIN
"""
from __future__ import annotations

import argparse, csv, glob, itertools, os, pickle, statistics, time
from pathlib import Path

import numpy as np

from mbiochain.bio_key import Params, enroll, pin_factor, recover
from mbiochain.fp_align import enroll_bits, load_sample, query_bits

PIN = "4821"


def load_all(root: str, cache: Path):
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    S, t = {}, []
    for f in sorted(glob.glob(os.path.join(root, "*DB*_B*", "*.tif"))):
        db = [p for p in f.split(os.sep)[-2].replace("-", "_").split("_") if p.startswith("DB")][0]
        finger, imp = os.path.basename(f)[:-4].split("_")
        t0 = time.perf_counter()
        S[(db, finger, int(imp))] = load_sample(f)
        t.append(time.perf_counter() - t0)
    pickle.dump((S, t), open(cache, "wb"))
    return S, t


def eer(g, i):
    ths = np.linspace(0, 0.6, 1201)
    frr = np.array([(g > x).mean() for x in ths]); far = np.array([(i <= x).mean() for x in ths])
    k = np.argmin(np.abs(frr - far))
    return 50 * (frr[k] + far[k]), ths[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fvc", required=True)
    ap.add_argument("--out", default="results_fvc")
    ap.add_argument("--ks", default="2,3,4,5,6,8")
    ap.add_argument("--multi", action="store_true", help="enroll with majority vote of impressions 1-3")
    ap.add_argument("--mode", default="mixed", choices=["mixed", "density"])
    a = ap.parse_args()
    import mbiochain.fp_align as FA
    FA.MODE = a.mode
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S, t_feat = load_all(a.fvc, out / "samples.pkl")
    fingers = sorted({k[:2] for k in S})
    print(f"{len(S)} images, {len(fingers)} fingers; feature extraction {1000*statistics.mean(t_feat):.0f} ms/image")

    enr_imps = [1, 2, 3] if a.multi else [1]
    gen_imps = [i for i in range(1, 9) if i not in enr_imps]
    ref, helpers, Q = {}, {}, {}
    t_align = []
    for f in fingers:
        b1, h = enroll_bits(S[f + (1,)])
        stack = [b1] + [query_bits(S[f + (i,)], h) for i in enr_imps[1:]]
        ref[f] = (np.mean(stack, 0) > 0.5).astype(np.uint8)
        helpers[f] = h
    for f in fingers:
        for i in gen_imps:
            t0 = time.perf_counter(); Q[("g", f, i)] = query_bits(S[f + (i,)], helpers[f]); t_align.append(time.perf_counter() - t0)
        for o in fingers:
            if o != f and o[0] == f[0]:
                Q[("i", f, o)] = query_bits(S[o + (1,)], helpers[f])
    g = np.array([np.mean(Q[k] != ref[k[1]]) for k in Q if k[0] == "g"])
    im = np.array([np.mean(Q[k] != ref[k[1]]) for k in Q if k[0] == "i"])
    e, thr = eer(g, im)
    print(f"Biometric baseline (Hamming distance): genuine {g.mean():.3f}, impostor {im.mean():.3f}, EER {e:.2f}%")
    per_db = {}
    for db in sorted({f[0] for f in fingers}):
        gd = np.array([np.mean(Q[k] != ref[k[1]]) for k in Q if k[0] == "g" and k[1][0] == db])
        idd = np.array([np.mean(Q[k] != ref[k[1]]) for k in Q if k[0] == "i" and k[1][0] == db])
        per_db[db] = (gd.mean(), idd.mean(), eer(gd, idd)[0])
        print(f"  {db}: genuine {gd.mean():.3f}  impostor {idd.mean():.3f}  EER {per_db[db][2]:.2f}%")
    with open(out / "hd_fvc.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["type", "db", "hd"])
        for k in Q:
            w.writerow(["genuine" if k[0] == "g" else "impostor", k[1][0], round(float(np.mean(Q[k] != ref[k[1]])), 4)])

    # fuzzy commitment, fingerprint only
    rows = []
    for k_rs in [int(x) for x in a.ks.split(",")]:
        p = Params(rs_k=k_rs, use_iris=False, use_finger=True, scrypt_n=2 ** 10)
        go = gt = io = it = 0
        for f in fingers:
            sk, pk, helper = enroll(None, ref[f], PIN, p)
            kp = pin_factor(PIN, helper.salt, p)
            for key in [k for k in Q if k[1] == f]:
                r = recover(None, Q[key], PIN, helper, k_pin=kp)
                if key[0] == "g":
                    gt += 1; go += int(r is not None and r[1].public_numbers() == pk.public_numbers())
                else:
                    it += 1; io += int(r is not None)
        rows.append(dict(rs=f"RS(16,{k_rs})", key_bits=7 * k_rs, t_symbols=(16 - k_rs) // 2,
                         genuine=gt, impostor=it, FRR=round(100 * (1 - go / gt), 2), FAR=round(100 * io / it, 2)))
        print(f"  RS(16,{k_rs:2d}) key {7*k_rs:3d} bits: FRR {rows[-1]['FRR']:6.2f}%  FAR {rows[-1]['FAR']:6.2f}%  ({gt} gen / {it} imp)")
    with open(out / "accuracy_fvc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary = dict(mode=a.mode, images=len(S), fingers=len(fingers), enrollment="majority 1-3" if a.multi else "impression 1",
                   feat_ms=round(1000 * statistics.mean(t_feat), 1), align_ms=round(1000 * statistics.mean(t_align), 1),
                   gen_mean=round(float(g.mean()), 4), imp_mean=round(float(im.mean()), 4), EER=round(e, 2), EER_thr=round(thr, 3),
                   align_helper_bytes=next(iter(helpers.values())).size_bytes())
    with open(out / "summary_fvc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
    print(summary)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        ax.hist(g, bins=40, range=(0, 0.6), alpha=0.7, color="#2a6fdb", label="genuine", density=True)
        ax.hist(im, bins=40, range=(0, 0.6), alpha=0.7, color="#d9822b", label="impostor", density=True)
        ax.axvline(15 / 64, color="k", ls="--", lw=0.8); ax.text(15 / 64 + 0.005, ax.get_ylim()[1] * 0.85, "Hadamard\nradius", fontsize=6)
        ax.set_xlabel("normalized Hamming distance"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(out / "hd_fvc.pdf"); fig.savefig(out / "hd_fvc.png", dpi=200); plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
