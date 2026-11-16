"""
MBioChain biometric key generation: multimodal (iris + fingerprint) fuzzy commitment
hardened with a PIN, producing a secp256k1 key pair.

Enrollment  Gen(w_iris, w_fp, PIN, epoch)        -> (sk, pk, helper data P)
Recovery    Rep(w_iris', w_fp', PIN, P)          -> (sk, pk) or failure

Templates are binary NumPy arrays (uint8, values 0/1):
  * iris  : IrisCode of IRIS_ROWS x IRIS_COLS bits (default 8 x 256 = 2048 bits);
            circular shifts along the angular axis (columns) compensate eye rotation.
  * finger: fixed-length binary fingerprint vector (default 1024 bits), e.g. a
            binarised Minutia Cylinder-Code or spectral-minutiae representation.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .ecc_codes import Hadamard, ReedSolomon

SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

IRIS_ROWS, IRIS_COLS = 8, 256
FP_BITS = 1024
BLOCK = 64  # Hadamard block length


@dataclass
class Params:
    rs_k: int = 20                 # RS message symbols (7 bits each) -> key entropy budget = 7*k bits
    use_iris: bool = True
    use_finger: bool = True
    permute: bool = True           # PIN-keyed bit permutation (cancelable / anti cross-matching)
    max_shift: int = 8             # iris rotation search range (columns)
    scrypt_n: int = 2 ** 15
    scrypt_r: int = 8
    scrypt_p: int = 1

    def n_bits(self) -> int:
        return (IRIS_ROWS * IRIS_COLS if self.use_iris else 0) + (FP_BITS if self.use_finger else 0)

    def rs_n(self) -> int:
        return self.n_bits() // BLOCK


@dataclass
class HelperData:
    """Public helper data. Stored off-chain (device / IPFS); only its hash goes on-chain."""
    version: int
    epoch: int
    salt: bytes
    offset: np.ndarray            # delta = C xor pi(w)
    tag: bytes                    # HMAC(K_bio, "check") to verify successful recovery
    params: Params = field(default_factory=Params)

    def digest(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.version.to_bytes(2, "big") + self.epoch.to_bytes(4, "big"))
        h.update(self.salt + np.packbits(self.offset).tobytes() + self.tag)
        return h.digest()

    def size_bytes(self) -> int:
        return 2 + 4 + len(self.salt) + len(np.packbits(self.offset)) + len(self.tag)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _hkdf(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def pin_factor(pin: str, salt: bytes, p: Params) -> bytes:
    """Memory-hard PIN stretching (scrypt, RFC 7914)."""
    return hashlib.scrypt(pin.encode(), salt=salt, n=p.scrypt_n, r=p.scrypt_r, p=p.scrypt_p,
                          maxmem=256 * 1024 * 1024, dklen=32)


def permutation(k_pin: bytes, salt: bytes, epoch: int, n: int) -> np.ndarray:
    seed = _hkdf(k_pin, salt, b"MBioChain|perm|" + epoch.to_bytes(4, "big"))
    return np.random.default_rng(int.from_bytes(seed, "big")).permutation(n)


def fuse(iris: np.ndarray | None, finger: np.ndarray | None, p: Params) -> np.ndarray:
    """Feature-level fusion by concatenation."""
    parts = []
    if p.use_iris:
        parts.append(np.asarray(iris, dtype=np.uint8).reshape(-1))
    if p.use_finger:
        parts.append(np.asarray(finger, dtype=np.uint8).reshape(-1))
    return np.concatenate(parts)


def symbols_to_bytes(symbols: list[int]) -> bytes:
    return bytes(symbols)  # each symbol < 128 fits in one byte


def derive_keypair(k_bio: bytes, k_pin: bytes, salt: bytes, epoch: int):
    """seed = HKDF(K_bio || K_pin) -> sk in [1, n-1] (64 bytes reduced mod n: bias < 2^-256)."""
    info = b"MBioChain|secp256k1|" + epoch.to_bytes(4, "big")
    okm = _hkdf(k_bio + k_pin, salt, info, length=64)
    d = int.from_bytes(okm, "big") % (SECP256K1_N - 1) + 1
    sk = ec.derive_private_key(d, ec.SECP256K1())
    return sk, sk.public_key()


# --------------------------------------------------------------------------
# Gen / Rep
# --------------------------------------------------------------------------


def enroll(iris: np.ndarray | None, finger: np.ndarray | None, pin: str,
           p: Params | None = None, epoch: int = 0):
    p = p or Params()
    n = p.rs_n()
    rs, had = ReedSolomon(n, p.rs_k, m=7), Hadamard(6)

    salt = os.urandom(16)
    k_pin = pin_factor(pin, salt, p)
    w = fuse(iris, finger, p)
    if p.permute:
        w = w[permutation(k_pin, salt, epoch, len(w))]

    s = [secrets.randbelow(128) for _ in range(p.rs_k)]       # random secret (7*k bits)
    codeword = had.encode(rs.encode(s))                          # n * 64 bits
    offset = codeword ^ w
    k_bio = hashlib.sha256(b"MBioChain|kbio|" + symbols_to_bytes(s)).digest()
    tag = hmac.new(k_bio, b"MBioChain|check|" + salt, hashlib.sha256).digest()

    sk, pk = derive_keypair(k_bio, k_pin, salt, epoch)
    helper = HelperData(version=1, epoch=epoch, salt=salt, offset=offset, tag=tag, params=p)
    return sk, pk, helper


def recover(iris: np.ndarray | None, finger: np.ndarray | None, pin: str, helper: HelperData,
            k_pin: bytes | None = None):
    """Returns (sk, pk, shift_used) or None. k_pin can be passed to skip scrypt in batch tests."""
    p = helper.params
    n = p.rs_n()
    rs, had = ReedSolomon(n, p.rs_k, m=7), Hadamard(6)
    k_pin = k_pin or pin_factor(pin, helper.salt, p)
    perm = permutation(k_pin, helper.salt, helper.epoch, p.n_bits()) if p.permute else None

    shifts = [0]
    if p.use_iris:
        for s_ in range(1, p.max_shift + 1):
            shifts += [s_, -s_]
    for sh in shifts:
        ir = None
        if p.use_iris:
            ir = np.roll(np.asarray(iris).reshape(IRIS_ROWS, IRIS_COLS), sh, axis=1)
        w = fuse(ir, finger, p)
        if perm is not None:
            w = w[perm]
        symbols = had.decode(helper.offset ^ w)
        s = rs.decode(symbols)
        if s is None:
            continue
        k_bio = hashlib.sha256(b"MBioChain|kbio|" + symbols_to_bytes(s)).digest()
        tag = hmac.new(k_bio, b"MBioChain|check|" + helper.salt, hashlib.sha256).digest()
        if hmac.compare_digest(tag, helper.tag):
            sk, pk = derive_keypair(k_bio, k_pin, helper.salt, helper.epoch)
            return sk, pk, sh
    return None
