"""
Iris image -> 2048-bit IrisCode (8 x 256) + noise mask, in pure OpenCV / NumPy (Daugman-style).

  1. pupil boundary   : dark-region threshold + largest round component + circle fit
  2. iris boundary    : radial intensity-gradient search on circles concentric with the pupil
                        (lateral sectors only, to avoid eyelids)
  3. normalisation    : Daugman rubber-sheet model -> R_N x A_N polar image (default 32 x 256)
  4. noise mask       : eyelids / eyelashes (very dark) and specular reflections (very bright)
  5. encoding         : 1-D log-Gabor filtering along the angular direction, phase quantised
                        to 2 bits (sign of real and imaginary parts) on 4 radial bands
                        -> rows 0-3 = real bits, rows 4-7 = imaginary bits, 256 angular columns.
     A rotation of the eye is a circular shift of the columns, compensated in bio_key.recover().

Works with near-infrared images (CASIA-Iris-Interval, IIT Delhi) and visible-light colour images
(UBIRIS.v1: segmentation on grey levels, encoding on the red channel, specular highlights inpainted).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

R_N, A_N = 32, 256          # normalised polar image (radial x angular)
BANDS = 4                   # radial bands -> 4 x 2 = 8 rows


@dataclass
class IrisSegmentation:
    pupil: tuple[float, float, float]   # (x, y, r)
    iris: tuple[float, float, float]


def remove_highlights(img: np.ndarray) -> np.ndarray:
    """Inpaint specular reflections (very bright spots, typically inside the pupil)."""
    thr = max(200, np.percentile(img, 99.5))
    m = (img >= thr).astype(np.uint8)
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA) if m.any() else img


def _ido(g: np.ndarray, centers: np.ndarray, radii: np.ndarray, angs: np.ndarray, sigma: float = 1.5):
    """Daugman's integro-differential operator: for every candidate centre, the radius that maximises
    the blurred radial derivative of the mean intensity on circular arcs. Returns (score, cx, cy, r)."""
    h, w = g.shape
    X = centers[:, 0, None, None] + radii[None, :, None] * np.cos(angs)[None, None, :]
    Y = centers[:, 1, None, None] + radii[None, :, None] * np.sin(angs)[None, None, :]
    vals = g[np.clip(Y.astype(int), 0, h - 1), np.clip(X.astype(int), 0, w - 1)].mean(-1)   # (N, K)
    d = np.diff(vals, axis=1)
    k = np.exp(-0.5 * (np.arange(-3, 4) / sigma) ** 2); k /= k.sum()
    d = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, d)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(d[i, j]), float(centers[i, 0]), float(centers[i, 1]), float(radii[j] + 0.5)


def _grid(cx, cy, half, step):
    o = np.arange(-half, half + 1e-9, step)
    gx, gy = np.meshgrid(cx + o, cy + o)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def segment(img: np.ndarray) -> IrisSegmentation:
    """Iris first (limbus = strongest circular edge, lateral arcs only to avoid eyelids),
    then pupil searched inside the iris only (eyelashes and eyebrows cannot be selected)."""
    g = cv2.GaussianBlur(remove_highlights(img), (5, 5), 0).astype(np.float32)
    h, w = g.shape
    lat = np.deg2rad(np.concatenate([np.arange(-35, 36, 5), np.arange(145, 216, 5)]))
    # 1) coarse limbus search on a grid around the image centre
    radii = np.arange(int(0.10 * min(h, w)), int(0.45 * min(h, w)), 2)
    _, cx, cy, R = _ido(g, _grid(w / 2, h / 2, 0.25 * min(h, w), 8), radii, lat)
    # refine
    _, cx, cy, R = _ido(g, _grid(cx, cy, 8, 2), np.arange(max(5, R - 12), R + 12, 1), lat)
    # 2) pupil inside the iris, full circles (pupil boundary is rarely occluded)
    full = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    pr_radii = np.arange(max(4, int(0.12 * R)), int(0.65 * R), 1)
    _, px, py, pr = _ido(g, _grid(cx, cy, 0.25 * R, 2), pr_radii, full)
    return IrisSegmentation((px, py, pr), (cx, cy, R))


def normalize(img: np.ndarray, seg: IrisSegmentation):
    """Rubber-sheet model: linear interpolation between pupil and iris boundaries."""
    (px, py, pr), (ix, iy, ir) = seg.pupil, seg.iris
    theta = np.linspace(0, 2 * np.pi, A_N, endpoint=False)
    rho = (np.arange(R_N) + 0.5) / R_N
    xp, yp = px + pr * np.cos(theta), py + pr * np.sin(theta)
    xi, yi = ix + ir * np.cos(theta), iy + ir * np.sin(theta)
    X = (1 - rho)[:, None] * xp[None] + rho[:, None] * xi[None]
    Y = (1 - rho)[:, None] * yp[None] + rho[:, None] * yi[None]
    polar = cv2.remap(img.astype(np.float32), X.astype(np.float32), Y.astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    inside = (X >= 0) & (X < img.shape[1]) & (Y >= 0) & (Y < img.shape[0])
    return polar, inside


def noise_mask(polar: np.ndarray, inside: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(polar[inside], [3, 99]) if inside.any() else (0, 255)
    valid = inside & (polar > max(lo, 0.35 * np.median(polar))) & (polar < max(hi, 235))
    return valid


def log_gabor_1d(n: int, wavelength: float = 18.0, sigma_on_f: float = 0.5) -> np.ndarray:
    f = np.fft.fftfreq(n)
    f0 = 1.0 / wavelength
    lg = np.zeros(n)
    pos = f > 0
    lg[pos] = np.exp(-(np.log(f[pos] / f0) ** 2) / (2 * np.log(sigma_on_f) ** 2))
    return lg


def encode(polar: np.ndarray, valid: np.ndarray):
    """-> code (8 x 256 bits), mask (8 x 256 bool)."""
    p = polar.copy()
    p[~valid] = np.mean(p[valid]) if valid.any() else 0
    bands = p.reshape(BANDS, R_N // BANDS, A_N).mean(1)          # 4 x 256
    vb = valid.reshape(BANDS, R_N // BANDS, A_N).mean(1) > 0.6
    G = log_gabor_1d(A_N)
    resp = np.fft.ifft(np.fft.fft(bands - bands.mean(1, keepdims=True), axis=1) * G[None], axis=1)
    code = np.concatenate([(resp.real > 0), (resp.imag > 0)]).astype(np.uint8)
    mask = np.concatenate([vb, vb])
    return code, mask


def iris_code(path: str):
    col = cv2.imread(path, cv2.IMREAD_COLOR)
    if col is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    is_colour = float(np.abs(col[..., 2].astype(int) - col[..., 0].astype(int)).mean()) > 8
    if max(img.shape) > 400:                                # e.g. UBIRIS 800x600 -> 400x300
        f = 400 / max(img.shape)
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        col = cv2.resize(col, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    seg = segment(img)
    tex = col[..., 2] if is_colour else img                 # red channel: best iris texture in visible light
    polar, inside = normalize(remove_highlights(tex), seg)
    code, mask = encode(polar, noise_mask(polar, inside))
    return code, mask, seg


def masked_hd(a, ma, b, mb, max_shift: int = 8) -> float:
    best = 1.0
    for s in range(-max_shift, max_shift + 1):
        bs, ms = np.roll(b, s, axis=1), np.roll(mb, s, axis=1)
        m = ma & ms
        if m.sum() > 0.3 * m.size:
            best = min(best, float(((a != bs) & m).sum() / m.sum()))
    return best
