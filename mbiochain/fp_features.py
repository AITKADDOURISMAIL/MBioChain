"""
Fingerprint image -> minutiae -> fixed-length binary vector (1024 bits).

Pipeline (pure OpenCV / scikit-image, no external SDK):
  1. segmentation (block variance)              -> foreground mask
  2. orientation field (smoothed gradients)
  3. contextual Gabor enhancement (16 orientations, fixed ridge frequency)
  4. binarisation + skeletonisation
  5. minutiae by crossing number (endings / bifurcations), border & cluster filtering
  6. spectral minutiae representation (Xu & Veldhuis, 2009): magnitude of the 2-D Fourier
     transform of the minutiae set sampled on a polar grid (R radial x A angular, angles in [0, pi)).
     The magnitude is translation invariant; a rotation of the finger becomes a circular shift
     along the angular axis, which is compensated at recovery time by a shift search.
  7. binarisation of each radial row against its median -> R x A bits (default 16 x 64 = 1024)
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

FP_R, FP_A = 16, 64           # 16 radial x 64 angular samples = 1024 bits


def segment(img: np.ndarray, block: int = 16, rel_thr: float = 0.35) -> np.ndarray:
    f = img.astype(np.float32)
    mean = cv2.blur(f, (block, block))
    sq = cv2.blur(f * f, (block, block))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    mask = (std > rel_thr * std.max()).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block, block))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    hull = cv2.convexHull(cv2.findNonZero(mask))
    full = np.zeros_like(mask)
    cv2.fillConvexPoly(full, hull, 1)
    return full


def orientation(f: np.ndarray, sigma: float = 7.0) -> np.ndarray:
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    return 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2   # ridge direction in [0, pi)


def enhance(img: np.ndarray, mask: np.ndarray, period: float = 9.0, n_orient: int = 16) -> np.ndarray:
    f = img.astype(np.float32)
    m = mask.astype(bool)
    f = (f - f[m].mean()) / (f[m].std() + 1e-6)
    f[~m] = 0
    theta = orientation(f)
    responses = []
    for k in range(n_orient):
        th = k * np.pi / n_orient
        # kernel oscillates across the ridges, i.e. along th + pi/2
        kern = cv2.getGaborKernel((25, 25), 4.0, th + np.pi / 2, period, 1.0, 0, ktype=cv2.CV_32F)
        kern -= kern.mean()
        responses.append(cv2.filter2D(f, cv2.CV_32F, kern))
    responses = np.stack(responses)
    idx = np.round(theta / (np.pi / n_orient)).astype(int) % n_orient
    out = np.take_along_axis(responses, idx[None], 0)[0]
    out[~m] = 0
    return out


def minutiae(img: np.ndarray, border: int = 14, min_dist: int = 8):
    mask = segment(img)
    enh = enhance(img, mask)
    ridges = (enh < 0) & mask.astype(bool)                       # ridges are dark
    ridges = cv2.morphologyEx(ridges.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    skel = skeletonize(ridges.astype(bool)).astype(np.uint8)
    # crossing number
    p = np.pad(skel, 1)
    nb = [p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, 0:-2], p[1:-1, 0:-2], p[0:-2, 0:-2]]
    cn = sum(np.abs(nb[i].astype(int) - nb[(i + 1) % 8].astype(int)) for i in range(8)) // 2
    inner = cv2.erode(mask, np.ones((2 * border + 1, 2 * border + 1), np.uint8)).astype(bool)
    ys, xs = np.nonzero((skel == 1) & ((cn == 1) | (cn == 3)) & inner)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    # remove clusters of minutiae (spurious pairs from breaks / bridges)
    if len(pts) > 1:
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        np.fill_diagonal(d, np.inf)
        pts = pts[(d >= min_dist).all(1)]
    return pts, mask


def spectral_minutiae(pts: np.ndarray, r: int = FP_R, a: int = FP_A,
                      lam_lo: float = 0.05, lam_hi: float = 0.45) -> np.ndarray:
    """|sum_j exp(-i w . x_j)| sampled on a polar grid; angles in [0, pi) (magnitude is symmetric)."""
    if len(pts) == 0:
        return np.zeros((r, a), np.float32)
    lam = np.linspace(lam_lo, lam_hi, r)
    ang = np.arange(a) * np.pi / a
    wx = lam[:, None] * np.cos(ang)[None]
    wy = lam[:, None] * np.sin(ang)[None]
    c = pts - pts.mean(0)
    phase = wx[..., None] * c[:, 0] + wy[..., None] * c[:, 1]          # (r, a, n)
    mag = np.abs(np.exp(-1j * phase).sum(-1)) / np.sqrt(len(pts))
    return mag.astype(np.float32)


def binarize(feat: np.ndarray) -> np.ndarray:
    return (feat > np.median(feat, axis=1, keepdims=True)).astype(np.uint8)


def fingerprint_bits(path: str) -> tuple[np.ndarray, int]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    pts, _ = minutiae(img)
    return binarize(spectral_minutiae(pts)), len(pts)
