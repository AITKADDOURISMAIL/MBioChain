"""
MBioChain experiments -> CSV files, figures and LaTeX table rows for the paper.

Usage
-----
  # pipeline check on synthetic data (NOT valid biometric results):
  python3 experiments.py --synthetic

  # real data (see mbiochain/data.py for the folder layout):
  python3 experiments.py --data /path/to/templates --subjects 100 --impostors 20

Outputs (in --out, default ./results):
  hd_distributions.csv / .pdf      genuine vs impostor Hamming distances      (Hamming distance figure)
  accuracy.csv / frr_vs_keylength.png     FRR / FAR vs key length for each modality  (Table "Key Regeneration Accuracy", FRR figure)
  timing.csv                       computation cost of each step              (Table "Computation Time")
  unlinkability.csv                cross-matching (decodability) attack       (Table "Decodability Cross-Matching Attack")
  tables.tex                       ready-to-paste LaTeX rows
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from mbiochain import ledger as L
from mbiochain.bio_key import (IRIS_COLS, IRIS_ROWS, Params, derive_keypair, enroll, fuse,
                               pin_factor, recover)
from mbiochain.data import FolderDB, SyntheticDB
from mbiochain.ecc_codes import Hadamard, ReedSolomon

PIN = "4821"
CONFIGS = {
    "Iris only": dict(use_iris=True, use_finger=False),
    "Fingerprint only": dict(use_iris=False, use_finger=True),
    "Iris + Fingerprint (proposed)": dict(use_iris=True, use_finger=True),
}


# ----------------------------------------------------------------------------- HD
def hd(a, b):
    return float(np.mean(a != b))


def iris_hd(a, b, max_shift=8):
    return min(hd(np.roll(a, s, axis=1), b) for s in range(-max_shift, max_shift + 1))


def hd_distributions(db, out: Path, n_imp: int):
    rows = []
    for s in db.subjects:
        ref_i, ref_f = db.iris[s][0], db.finger[s][0]
        for x_i, x_f in zip(db.iris[s][1:], db.finger[s][1:]):
            di, df = iris_hd(x_i, ref_i), hd(x_f, ref_f)
            rows.append(("genuine", di, df, (2048 * di + 1024 * df) / 3072))
        others = [o for o in db.subjects if o != s][:n_imp]
        for o in others:
            di, df = iris_hd(db.iris[o][0], ref_i), hd(db.finger[o][0], ref_f)
            rows.append(("impostor", di, df, (2048 * di + 1024 * df) / 3072))
    with open(out / "hd_distributions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "hd_iris", "hd_finger", "hd_fused"])
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=False)
        for ax, col, name in zip(axes, [1, 2, 3], ["Iris", "Fingerprint", "Fused"]):
            g = [r[col] for r in rows if r[0] == "genuine"]
            i = [r[col] for r in rows if r[0] == "impostor"]
            ax.hist(g, bins=40, range=(0, 0.6), alpha=0.7, label="genuine", color="#2a6fdb")
            ax.hist(i, bins=40, range=(0, 0.6), alpha=0.7, label="impostor", color="#d9822b")
            ax.set_title(name)
            ax.set_xlabel("normalized Hamming distance")
        axes[0].set_ylabel("count")
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "hd_distributions.png", dpi=300)
        fig.savefig(out / "hd_distributions.pdf")
        plt.close(fig)
    except ImportError:
        pass
    return rows


# ------------------------------------------------------------------------ accuracy
def accuracy(db, out: Path, ks: list[int], n_imp: int):
    results = []
    for cname, flags in CONFIGS.items():
        for k in ks:
            p = Params(rs_k=k, scrypt_n=2 ** 10, **flags)   # small scrypt: does not affect accuracy
            if k >= p.rs_n():
                continue
            gen_ok = gen_tot = imp_ok = imp_tot = 0
            for s in db.subjects:
                sk, pk, helper = enroll(db.iris[s][0], db.finger[s][0], PIN, p)
                k_pin = pin_factor(PIN, helper.salt, p)
                for x_i, x_f in zip(db.iris[s][1:], db.finger[s][1:]):
                    r = recover(x_i, x_f, PIN, helper, k_pin=k_pin)
                    gen_tot += 1
                    gen_ok += int(r is not None and r[1].public_numbers() == pk.public_numbers())
                # worst case: attacker knows the PIN and presents another person's biometrics
                for o in [o for o in db.subjects if o != s][:n_imp]:
                    r = recover(db.iris[o][0], db.finger[o][0], PIN, helper, k_pin=k_pin)
                    imp_tot += 1
                    imp_ok += int(r is not None)
            frr = 100 * (1 - gen_ok / gen_tot)
            far = 100 * imp_ok / imp_tot
            n = p.rs_n()
            row = dict(config=cname, rs_n=n, rs_k=k, key_bits=7 * k, t_symbols=(n - k) // 2,
                       genuine_trials=gen_tot, impostor_trials=imp_tot,
                       FRR_percent=round(frr, 3), FAR_percent=round(far, 4), GAR_percent=round(100 - frr, 3))
            results.append(row)
            print(f"  {cname:32s} RS({n},{k}) key={7*k:3d} bits  FRR={frr:6.2f}%  FAR={far:.4f}%")
    with open(out / "accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3))
        for cname, mk in zip(CONFIGS, ["o", "s", "^"]):
            rr = [r for r in results if r["config"] == cname]
            ax.plot([r["key_bits"] for r in rr], [r["FRR_percent"] for r in rr], marker=mk, label=cname)
        ax.set_xlabel("biometric key length (bits)")
        ax.set_ylabel("FRR (%)")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "frr_vs_keylength.png", dpi=300)
        fig.savefig(out / "frr_vs_keylength.pdf")
        plt.close(fig)
    except ImportError:
        pass
    return results


# -------------------------------------------------------------------------- timing
def timing(db, out: Path, reps: int):
    p = Params()  # full scrypt N = 2^15
    s = db.subjects[0]
    iris0, fp0, iris1, fp1 = db.iris[s][0], db.finger[s][0], db.iris[s][1], db.finger[s][1]
    meas = {k: [] for k in ["PIN stretching (scrypt)", "Enrollment Gen (total)", "Recovery Rep (total)",
                            "Key derivation (HKDF + EC scalar mult.)", "ECDSA sign", "ECDSA verify",
                            "Register tx validation", "Block sealing (1 tx)"]}
    ledger = L.Ledger()
    guardians = [L.ec.generate_private_key(L.ec.SECP256K1()) for _ in range(3)]
    for i in range(reps):
        salt = np.random.bytes(16)
        t = time.perf_counter(); kp = pin_factor(PIN, salt, p); meas["PIN stretching (scrypt)"].append(time.perf_counter() - t)
        t = time.perf_counter(); sk, pk, helper = enroll(iris0, fp0, PIN, p); meas["Enrollment Gen (total)"].append(time.perf_counter() - t)
        t = time.perf_counter(); r = recover(iris1, fp1, PIN, helper); meas["Recovery Rep (total)"].append(time.perf_counter() - t)
        t = time.perf_counter(); derive_keypair(b"x" * 32, kp, salt, 0); meas["Key derivation (HKDF + EC scalar mult.)"].append(time.perf_counter() - t)
        msg = b"challenge" + salt
        t = time.perf_counter(); sig = L.sign(sk, msg); meas["ECDSA sign"].append(time.perf_counter() - t)
        t = time.perf_counter(); L.verify(pk, sig, msg); meas["ECDSA verify"].append(time.perf_counter() - t)
        tx = L.make_register(f"did:mbio:{i}", sk, helper.digest(), 0, guardians, 2)
        t = time.perf_counter(); ledger.submit(tx); meas["Register tx validation"].append(time.perf_counter() - t)
        t = time.perf_counter(); ledger.seal(); meas["Block sealing (1 tx)"].append(time.perf_counter() - t)
    rows = []
    for k, v in meas.items():
        ms = [x * 1000 for x in v]
        rows.append(dict(operation=k, mean_ms=round(statistics.mean(ms), 3),
                         std_ms=round(statistics.stdev(ms) if len(ms) > 1 else 0.0, 3)))
        print(f"  {k:42s} {rows[-1]['mean_ms']:9.3f} ms ± {rows[-1]['std_ms']:.3f}")
    rows.append(dict(operation="Helper data size (bytes)", mean_ms=helper.size_bytes(), std_ms=0))
    with open(out / "timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["operation", "mean_ms", "std_ms"])
        w.writeheader()
        w.writerows(rows)
    assert ledger.verify_chain()
    return rows


# ------------------------------------------------------------------- unlinkability
def _decodability_link(h1, h2) -> bool:
    """Cross-matching (decodability) attack: the concatenated code is GF(2)-linear, so
    d1 xor d2 = (c1 xor c2) xor (w1 xor w2) is decodable iff w1 and w2 are close."""
    p = h1.params
    rs, had = ReedSolomon(p.rs_n(), p.rs_k, m=7), Hadamard(6)
    return rs.decode(had.decode(h1.offset ^ h2.offset)) is not None


def unlinkability(db, out: Path, n_users: int, rs_k: int = 8):
    """Two helper data records of the same person (two epochs, or two services) must not be linkable.
    Mated (same capture)  = the same reference template enrolled twice (e.g. re-enrollment from a stored template)
    Mated (fresh capture) = two different captures of the same subject
    Non-mated             = two different subjects
    Without permutation the decodability attack can link mated pairs; with the PIN/epoch-keyed
    permutation the link rate should drop to the non-mated (chance) level."""
    rows = []
    subj = db.subjects[:n_users]
    for permute in (False, True):
        p = Params(rs_k=rs_k, scrypt_n=2 ** 10, permute=permute)
        same = fresh = nonmated = 0
        for idx, s in enumerate(subj):
            _, _, h1 = enroll(db.iris[s][0], db.finger[s][0], PIN, p, epoch=0)
            _, _, h1b = enroll(db.iris[s][0], db.finger[s][0], PIN, p, epoch=1)
            _, _, h2 = enroll(db.iris[s][1], db.finger[s][1], PIN, p, epoch=1)
            o = subj[(idx + 1) % len(subj)]
            _, _, h3 = enroll(db.iris[o][1], db.finger[o][1], PIN, p, epoch=1)
            same += _decodability_link(h1, h1b)
            fresh += _decodability_link(h1, h2)
            nonmated += _decodability_link(h1, h3)
        n = len(subj)
        rows.append(dict(permutation=permute, pairs=n, rs=f"RS({p.rs_n()},{rs_k})",
                         mated_same_capture_link_percent=round(100 * same / n, 2),
                         mated_fresh_capture_link_percent=round(100 * fresh / n, 2),
                         nonmated_link_percent=round(100 * nonmated / n, 2)))
        r = rows[-1]
        print(f"  permutation={permute!s:5s}  mated(same capture): {r['mated_same_capture_link_percent']:6.2f}%"
              f"  mated(fresh capture): {r['mated_fresh_capture_link_percent']:6.2f}%"
              f"  non-mated: {r['nonmated_link_percent']:6.2f}%")
    with open(out / "unlinkability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------- latex rows
def write_tables(out: Path, acc, tim, unl):
    lines = ["% ---- Table 'Key Regeneration Accuracy' (label tab:acc) ----"]
    for r in acc:
        lines.append(f"{r['config']} & RS({r['rs_n']},{r['rs_k']}) & {r['key_bits']} & "
                     f"{r['FRR_percent']:.2f} & {r['FAR_percent']:.3f} \\\\")
    lines.append("% ---- Table 'Computation Time' (label tab:time) ----")
    for r in tim:
        lines.append(f"{r['operation']} & {r['mean_ms']} & {r['std_ms']} \\\\")
    lines.append("% ---- Table 'Decodability Cross-Matching Attack' (label tab:unlink) ----")
    for r in unl:
        lines.append(f"{'with' if r['permutation'] else 'without'} permutation & "
                     f"{r['mated_same_capture_link_percent']:.2f} & {r['mated_fresh_capture_link_percent']:.2f} & "
                     f"{r['nonmated_link_percent']:.2f} \\\\")
    (out / "tables.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data", type=str)
    ap.add_argument("--subjects", type=int, default=50)
    ap.add_argument("--impostors", type=int, default=10, help="impostor attempts per enrolled user")
    ap.add_argument("--ks", type=str, default="4,8,12,16,20,24,28,32")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", type=str, default="results")
    a = ap.parse_args()
    if not a.synthetic and not a.data:
        ap.error("choose --synthetic or --data PATH")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    db = SyntheticDB(n_subjects=a.subjects) if a.synthetic else FolderDB(a.data, a.subjects)
    print(f"Database: {'SYNTHETIC (pipeline check only)' if a.synthetic else a.data} - {len(db.subjects)} subjects")
    print("[1/4] Hamming distance distributions"); hd_distributions(db, out, a.impostors)
    print("[2/4] FRR / FAR"); acc = accuracy(db, out, [int(x) for x in a.ks.split(",")], a.impostors)
    print("[3/4] Timing"); tim = timing(db, out, a.reps)
    print("[4/4] Unlinkability"); unl = unlinkability(db, out, min(20, len(db.subjects)))
    write_tables(out, acc, tim, unl)
    print(f"Done. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
