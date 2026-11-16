"""
Biometric template sources.

1) SyntheticDB  : synthetic binary templates for debugging the pipeline ONLY.
                  Numbers obtained with it must NOT be reported as biometric results.
2) FolderDB     : real templates extracted by the user, stored as

        <root>/iris/<subject_id>/<sample>.npy   (or .png)  -> 8 x 256 bits (any shape with 2048 bits)
        <root>/finger/<subject_id>/<sample>.npy (or .png)  -> 1024 bits

   A *virtual multimodal* database is built by pairing iris subject i with fingerprint
   subject i (common practice, e.g. CASIA-IrisV4 + FVC2002/2004), see Nagar et al. 2012.
   Subjects are matched by sorted order of folder names; samples by sorted order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .bio_key import FP_BITS, IRIS_COLS, IRIS_ROWS


class SyntheticDB:
    def __init__(self, n_subjects=50, n_samples=6, iris_ber=(0.10, 0.03), fp_ber=(0.20, 0.05),
                 max_rot=4, occlusion=0.06, seed=1):
        rng = np.random.default_rng(seed)
        self.subjects = list(range(n_subjects))
        self.iris, self.finger = {}, {}
        for s in self.subjects:
            iris_ref = rng.integers(0, 2, (IRIS_ROWS, IRIS_COLS), dtype=np.uint8)
            fp_ref = rng.integers(0, 2, FP_BITS, dtype=np.uint8)
            self.iris[s], self.finger[s] = [], []
            for _ in range(n_samples):
                # iris: rotation + random bit noise + occluded band (eyelid / eyelash)
                ber = float(np.clip(rng.normal(*iris_ber), 0.0, 0.5))
                x = iris_ref ^ (rng.random(iris_ref.shape) < ber).astype(np.uint8)
                o = int(rng.uniform(0, occlusion) * IRIS_COLS)
                if o:
                    c0 = rng.integers(0, IRIS_COLS - o)
                    x[:2, c0:c0 + o] = rng.integers(0, 2, (2, o))
                x = np.roll(x, int(rng.integers(-max_rot, max_rot + 1)), axis=1)
                self.iris[s].append(x)
                ber = float(np.clip(rng.normal(*fp_ber), 0.0, 0.5))
                self.finger[s].append(fp_ref ^ (rng.random(FP_BITS) < ber).astype(np.uint8))


def _load_bits(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        a = np.load(path)
    else:  # image: any non-zero pixel = 1
        import matplotlib.image as mpimg
        a = mpimg.imread(path)
        if a.ndim == 3:
            a = a[..., 0]
        a = (a > 0.5 * a.max()).astype(np.uint8)
    return (np.asarray(a).reshape(-1) > 0).astype(np.uint8)


class FolderDB:
    def __init__(self, root: str, n_subjects: int | None = None):
        root = Path(root)
        iris_dirs = sorted(d for d in (root / "iris").iterdir() if d.is_dir())
        fp_dirs = sorted(d for d in (root / "finger").iterdir() if d.is_dir())
        n = min(len(iris_dirs), len(fp_dirs))
        if n_subjects:
            n = min(n, n_subjects)
        self.subjects = list(range(n))
        self.iris, self.finger = {}, {}
        for s in self.subjects:
            ifiles = sorted(p for p in iris_dirs[s].iterdir() if p.suffix in (".npy", ".png", ".bmp"))
            ffiles = sorted(p for p in fp_dirs[s].iterdir() if p.suffix in (".npy", ".png", ".bmp"))
            m = min(len(ifiles), len(ffiles))
            self.iris[s] = [self._iris(_load_bits(p)) for p in ifiles[:m]]
            self.finger[s] = [self._fp(_load_bits(p)) for p in ffiles[:m]]

    @staticmethod
    def _iris(bits):
        if bits.size != IRIS_ROWS * IRIS_COLS:
            raise ValueError(f"iris template must have {IRIS_ROWS*IRIS_COLS} bits, got {bits.size}")
        return bits.reshape(IRIS_ROWS, IRIS_COLS)

    @staticmethod
    def _fp(bits):
        if bits.size != FP_BITS:
            raise ValueError(f"fingerprint template must have {FP_BITS} bits, got {bits.size}")
        return bits
