"""
Aligned fixed-length binary fingerprint representation (used for the FVC2002 experiments).

Enrollment stores, in the public helper data, a coarse *orientation field* of the enrollment
impression (8x8-pixel blocks, 4-bit quantised angle + foreground bit). At recovery, the query
orientation field is registered against it (rotation search + FFT cross-correlation for the
translation), as done with alignment helper data in fingerprint fuzzy vaults (Nandakumar et al.,
2007). No minutia is stored.

Binary vector (FP_BITS = 1024) computed in the enrollment frame:
  * 512 bits : Gaussian minutiae-density map on a 32 x 32 grid (step 9 px, sigma 20 px), thresholded
               at its median, checkerboard-subsampled
  * 512 bits : local ridge orientation on a 16 x 16 grid, 2-bit Gray code of the doubled angle
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fp_features import minutiae, segment

BS = 8                        # orientation block size (pixels)
DENS_G, DENS_STEP, DENS_SIGMA = 32, 9, 20.0
ORI_G, ORI_STEP = 16, 18
MODE = "mixed"                # "mixed" = 512 density + 512 orientation bits ; "density" = 1024 density bits
GRAY = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.uint8)


@dataclass
class FingerSample:
    pts: np.ndarray               # minutiae (x, y) in pixels
    field: np.ndarray             # complex doubled-angle orientation field, per 8x8 block
    mask: np.ndarray              # foreground mask (pixels)


@dataclass
class AlignHelper:
    q_angle: np.ndarray           # uint8, 4-bit quantised doubled angle per block
    fg: np.ndarray                # bool, foreground blocks
    center: tuple[float, float]   # grid centre (pixels, enrollment frame)

    def field(self) -> np.ndarray:
        return self.fg * np.exp(1j * (self.q_angle.astype(np.float32) + 0.5) * (2 * np.pi / 16))

    def size_bytes(self) -> int:
        return int(np.ceil(self.q_angle.size * 5 / 8)) + 4


def orientation_field(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    m = mask.astype(bool)
    f = (f - f[m].mean()) / (f[m].std() + 1e-6)
    f[~m] = 0
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    c = (gx + 1j * gy) ** 2
    c = cv2.GaussianBlur(c.real, (0, 0), 6) + 1j * cv2.GaussianBlur(c.imag, (0, 0), 6)
    c = c[BS // 2::BS, BS // 2::BS] * mask[BS // 2::BS, BS // 2::BS]
    return c / (np.abs(c).max() + 1e-9)


def load_sample(path: str) -> FingerSample:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    pts, mask = minutiae(img)
    return FingerSample(pts, orientation_field(img, mask), mask)


def _rotate_field(c: np.ndarray, ang: float) -> np.ndarray:
    h, w = c.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), np.rad2deg(ang), 1.0)
    re = cv2.warpAffine(c.real.astype(np.float32), M, (w, h))
    im = cv2.warpAffine(c.imag.astype(np.float32), M, (w, h))
    return (re + 1j * im) * np.exp(-2j * ang)


def register(E: np.ndarray, Q: np.ndarray, max_deg: int = 40, step_deg: int = 2):
    """Rigid registration of query field Q onto enrollment field E -> (angle, dx_blocks, dy_blocks)."""
    H = max(E.shape[0], Q.shape[0]) * 2
    W = max(E.shape[1], Q.shape[1]) * 2
    FE = np.fft.fft2(E, (H, W))
    best = (-np.inf, 0.0, 0, 0)
    for ang in np.deg2rad(np.arange(-max_deg, max_deg + 1, step_deg)):
        corr = np.fft.ifft2(FE * np.conj(np.fft.fft2(_rotate_field(Q, ang), (H, W)))).real
        iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
        if corr[iy, ix] > best[0]:
            best = (corr[iy, ix], ang, ix if ix < W // 2 else ix - W, iy if iy < H // 2 else iy - H)
    return best[1:]


def _to_enroll_frame(pts, ang, dx, dy, qshape):
    h, w = qshape
    cx, cy = w * BS / 2, h * BS / 2
    R = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    return (pts - [cx, cy]) @ R.T + [cx, cy] + [dx * BS, dy * BS]


def _grid(center, G, step):
    off = (np.arange(G) - (G - 1) / 2) * step
    gx, gy = np.meshgrid(center[0] + off, center[1] + off)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def _density_bits_full(pts_e, center):
    """512 density bits: 32x32 grid subsampled on a checkerboard (keeps the full spatial extent)."""
    grid = _grid(center, DENS_G, DENS_STEP)
    D = np.exp(-((grid[:, None] - pts_e[None]) ** 2).sum(-1) / (2 * DENS_SIGMA ** 2)).sum(1) if len(pts_e) else np.zeros(len(grid))
    bits = (D > np.median(D)).astype(np.uint8).reshape(DENS_G, DENS_G)
    if MODE == "density":
        return bits.ravel()
    checker = (np.add.outer(np.arange(DENS_G), np.arange(DENS_G)) % 2 == 0)
    return bits[checker]


def _orient_bits(field, ang, dx, dy, center, qshape):
    h, w = qshape
    cx, cy = w * BS / 2, h * BS / 2
    grid = _grid(center, ORI_G, ORI_STEP)
    R = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    q = (grid - [dx * BS, dy * BS] - [cx, cy]) @ R + [cx, cy]
    ix = np.clip((q[:, 0] / BS).astype(int), 0, w - 1)
    iy = np.clip((q[:, 1] / BS).astype(int), 0, h - 1)
    c = field[iy, ix] * np.exp(-2j * ang)
    b = np.floor((np.angle(c) % (2 * np.pi)) / (2 * np.pi) * 4).astype(int) % 4
    return GRAY[b].ravel()


def enroll_bits(s: FingerSample) -> tuple[np.ndarray, AlignHelper]:
    ys, xs = np.nonzero(s.mask)
    center = (float(xs.mean()), float(ys.mean()))
    q = np.floor((np.angle(s.field) % (2 * np.pi)) / (2 * np.pi) * 16).astype(np.uint8)
    helper = AlignHelper(q, np.abs(s.field) > 0.05, center)
    parts = [_density_bits_full(s.pts, center)]
    if MODE == "mixed":
        parts.append(_orient_bits(s.field, 0.0, 0, 0, center, s.field.shape))
    return np.concatenate(parts), helper


def query_bits(s: FingerSample, helper: AlignHelper) -> np.ndarray:
    ang, dx, dy = register(helper.field(), s.field)
    pts_e = _to_enroll_frame(s.pts, ang, dx, dy, s.field.shape)
    parts = [_density_bits_full(pts_e, helper.center)]
    if MODE == "mixed":
        parts.append(_orient_bits(s.field, ang, dx, dy, helper.center, s.field.shape))
    return np.concatenate(parts)
