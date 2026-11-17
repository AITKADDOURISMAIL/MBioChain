"""
Multimodal experiments: real iris (IIT Delhi v1.0 or CASIA-Iris-Interval) + real fingerprints (FVC2002 set B).

  python3 experiments_multimodal.py --iris DOSSIER_IRIS --fvc .\\fvc --out results_multi --runs 10
  UBIRIS.v1 : --iris .\\iris_ubiris\\UBIRIS_800_600\\Sessao_1

Virtual subjects: iris subject i (sorted folders, one eye) is paired with finger i (FVC2002 DB1..DB4).
  * enrollment : sample 1 of the iris + impression 1 of the finger
  * genuine    : sample j of both (j = 2..M)
  * impostor   : sample 1 of every other virtual subject (fingerprint registered on the target's helper),
                 correct PIN (worst case)
Outputs: hd_multi.csv/.pdf, accuracy_multi.csv, unlinkability_multi.csv, summary_multi.csv, frr_multi.pdf
"""
from __future__ import annotations

import argparse, csv, glob, itertools, os, pickle, re, time
from pathlib import Path

import numpy as np

from mbiochain.bio_key import Params, enroll, pin_factor, recover
from mbiochain.ecc_codes import Hadamard, ReedSolomon
from mbiochain.fp_align import enroll_bits, query_bits
from mbiochain.iris_features import iris_code, masked_hd
from experiments_fvc import load_all as load_fingers

PIN = "4821"
CONFIGS = {"Iris only": (True, False), "Fingerprint only": (False, True), "Iris + Fingerprint": (True, True)}


def iris_files(root: str, eye: str):
    """One list of image paths per subject folder, one eye (L/R) when the file names tell it."""
    subj = []
    natural = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]
    for d in sorted((p for p in Path(root).rglob("*") if p.is_dir()), key=natural):
        files = sorted((f for f in d.iterdir() if f.suffix.lower() in (".bmp", ".jpg", ".jpeg", ".png")), key=natural)
        if not files:
            continue
        sel = [f for f in files if re.search(rf"(^|[_\d]){eye}(\d|\.|_|$)", f.stem, re.I)] or files
        subj.append((d.name, sel))
    # CASIA stores eyes in sub-folders YYY/L, YYY/R: keep only the requested eye folder
    if any(n.upper() in ("L", "R") for n, _ in subj):
        subj = [(n, f) for n, f in subj if n.upper() == eye.upper()]
    return [f for _, f in subj if len(f) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iris", required=True)
    ap.add_argument("--fvc", required=True)
    ap.add_argument("--out", default="results_multi")
    ap.add_argument("--eye", default="L")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--ks", default="8,12,16,20,24,28")
    ap.add_argument("--unlink-k", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    S, _ = load_fingers(a.fvc, out / "samples.pkl")
    fingers = sorted({k[:2] for k in S})
    ifiles = iris_files(a.iris, a.eye)
    n = min(len(fingers), len(ifiles))
    M = min(5, min(len(f) for f in ifiles[:n]), 8)
    print(f"{n} virtual subjects (iris folders: {len(ifiles)}, fingers: {len(fingers)}), {M} samples each")

    cache = out / "iris.pkl"
    if cache.exists():
        I, t_iris = pickle.load(open(cache, "rb"))
    else:
        I, t_iris = {}, []
        for s in range(n):
            for j in range(M):
                t0 = time.perf_counter(); I[(s, j)] = iris_code(str(ifiles[s][j])); t_iris.append(time.perf_counter() - t0)
        pickle.dump((I, t_iris), open(cache, "wb"))
    print(f"iris coding: {1000*np.mean(t_iris):.0f} ms/image")

    # fingerprint vectors: reference + queries registered on the enrollment helper of the target subject
    F_ref, F_help = {}, {}
    for s in range(n):
        F_ref[s], F_help[s] = enroll_bits(S[fingers[s] + (1,)])
    Fq = {}
    for s in range(n):
        for j in range(1, M):
            Fq[("g", s, j)] = query_bits(S[fingers[s] + (j + 1,)], F_help[s])
        for o in range(n):
            if o != s:
                Fq[("i", s, o)] = query_bits(S[fingers[o] + (1,)], F_help[s])

    def iris_of(key):            # (code, mask) of the presented iris
        return I[(key[1], key[2])] if key[0] == "g" else I[(key[2], 0)]

    keys = [("g", s, j) for s in range(n) for j in range(1, M)] + [("i", s, o) for s in range(n) for o in range(n) if o != s]

    # Hamming distances
    rows = []
    for k in keys:
        s = k[1]
        hi = masked_hd(*I[(s, 0)][:2], *iris_of(k)[:2])
        hf = float(np.mean(Fq[k] != F_ref[s]))
        rows.append(("genuine" if k[0] == "g" else "impostor", hi, hf, (2048 * hi + 1024 * hf) / 3072))
    with open(out / "hd_multi.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["type", "hd_iris", "hd_finger", "hd_fused"]); w.writerows(rows)
    for c, name in [(1, "iris"), (2, "finger"), (3, "fused")]:
        g = [r[c] for r in rows if r[0] == "genuine"]; i = [r[c] for r in rows if r[0] == "impostor"]
        print(f"  HD {name:6s}: genuine {np.mean(g):.3f}  impostor {np.mean(i):.3f}")

    # key regeneration, repeated runs
    acc = {}
    for run in range(a.runs):
        for cname, (ui, uf) in CONFIGS.items():
            for k_rs in [int(x) for x in a.ks.split(",")]:
                p = Params(rs_k=k_rs, use_iris=ui, use_finger=uf, scrypt_n=2 ** 10)
                if k_rs >= p.rs_n() - 1:
                    continue
                go = gt = io = it = 0
                for s in range(n):
                    sk, pk, helper = enroll(I[(s, 0)][0], F_ref[s], PIN, p)
                    kp = pin_factor(PIN, helper.salt, p)
                    for key in [k for k in keys if k[1] == s]:
                        r = recover(iris_of(key)[0], Fq[key], PIN, helper, k_pin=kp)
                        if key[0] == "g":
                            gt += 1; go += int(r is not None and r[1].public_numbers() == pk.public_numbers())
                        else:
                            it += 1; io += int(r is not None)
                acc.setdefault((cname, k_rs, p.rs_n()), []).append((100 * (1 - go / gt), 100 * io / it))
        print(f"  run {run + 1}/{a.runs} done")
    ms = lambda v: (round(float(np.mean(v)), 2), round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 2))
    arows = []
    for (cname, k_rs, nn), v in acc.items():
        (fr, frs), (fa, fas) = ms([x[0] for x in v]), ms([x[1] for x in v])
        arows.append(dict(config=cname, rs=f"RS({nn},{k_rs})", key_bits=7 * k_rs, FRR_mean=fr, FRR_std=frs, FAR_mean=fa, FAR_std=fas))
        print(f"  {cname:20s} RS({nn},{k_rs:2d}) {7*k_rs:3d} bits: FRR {fr:6.2f} +/- {frs:4.2f}%  FAR {fa:6.2f} +/- {fas:4.2f}%")
    with open(out / "accuracy_multi.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arows[0])); w.writeheader(); w.writerows(arows)

    # cross-matching attack on fused helper data
    def linked(h1, h2):
        pp = h1.params
        return ReedSolomon(pp.rs_n(), pp.rs_k, m=7).decode(Hadamard(6).decode(h1.offset ^ h2.offset)) is not None
    urows = []
    for permute in (False, True):
        p = Params(rs_k=a.unlink_k, scrypt_n=2 ** 10, permute=permute)
        same = fresh = non = ns = nf = nn_ = 0
        for s in range(n):
            _, _, h1 = enroll(I[(s, 0)][0], F_ref[s], PIN, p, epoch=0)
            _, _, h1b = enroll(I[(s, 0)][0], F_ref[s], PIN, p, epoch=1)
            same += linked(h1, h1b); ns += 1
            for key in [k for k in keys if k[1] == s]:
                _, _, h2 = enroll(iris_of(key)[0], Fq[key], PIN, p, epoch=1)
                if key[0] == "g":
                    fresh += linked(h1, h2); nf += 1
                else:
                    non += linked(h1, h2); nn_ += 1
        urows.append(dict(permutation=permute, rs=f"RS(48,{a.unlink_k})", same=round(100 * same / ns, 2),
                          fresh=round(100 * fresh / nf, 2), nonmated=round(100 * non / nn_, 2)))
        print(f"  Unlinkability permutation={permute!s:5s}: same {urows[-1]['same']:6.2f}%  fresh {urows[-1]['fresh']:6.2f}%  non-mated {urows[-1]['nonmated']:6.2f}%")
    with open(out / "unlinkability_multi.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(urows[0])); w.writeheader(); w.writerows(urows)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
        for ax, c, name in zip(axes, [1, 2, 3], ["Iris", "Fingerprint", "Fused"]):
            ax.hist([r[c] for r in rows if r[0] == "genuine"], bins=30, range=(0, 0.6), alpha=0.7, density=True, color="#2a6fdb", label="genuine")
            ax.hist([r[c] for r in rows if r[0] == "impostor"], bins=30, range=(0, 0.6), alpha=0.7, density=True, color="#d9822b", label="impostor")
            ax.axvline(15 / 64, color="k", ls="--", lw=0.8); ax.set_title(name, fontsize=9); ax.set_xlabel("normalized HD", fontsize=8)
        axes[0].set_ylabel("density"); axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(out / "hd_distributions.pdf"); fig.savefig(out / "hd_distributions.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for cname, mk in zip(CONFIGS, ["o", "s", "^"]):
            rr = [r for r in arows if r["config"] == cname]
            ax.errorbar([r["key_bits"] for r in rr], [r["FRR_mean"] for r in rr], [r["FRR_std"] for r in rr], marker=mk, capsize=2, label=cname)
        ax.set_xlabel("biometric secret length (bits)"); ax.set_ylabel("FRR (%)"); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(out / "frr_vs_keylength.pdf"); fig.savefig(out / "frr_vs_keylength.png", dpi=200); plt.close(fig)
    except ImportError:
        pass
    print(f"Done. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
