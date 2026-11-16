"""
Error-correcting codes used by the multimodal fuzzy commitment.

Two-layer concatenated code (in the spirit of Hao, Anderson & Daugman, 2006):
  * inner code : augmented Hadamard / first-order Reed-Muller RM(1,6)
                 7 information bits -> 64-bit block, corrects up to 15 bit errors
                 (handles *random* bit noise inside a block)
  * outer code : Reed-Solomon RS(n, k) over GF(2^7)
                 corrects up to t = (n-k)//2 wrong 7-bit symbols
                 (handles *blocks* that the inner code could not correct)

Pure Python / NumPy, no external dependency.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Galois field GF(2^m)
# --------------------------------------------------------------------------


class GF2m:
    """Arithmetic in GF(2^m) using log / antilog tables."""

    PRIMITIVE_POLYS = {4: 0x13, 5: 0x25, 6: 0x43, 7: 0x89, 8: 0x11D}

    def __init__(self, m: int = 7):
        if m not in self.PRIMITIVE_POLYS:
            raise ValueError(f"unsupported field size m={m}")
        self.m = m
        self.q = 1 << m
        self.order = self.q - 1
        prim = self.PRIMITIVE_POLYS[m]
        self.exp = [0] * (2 * self.order)
        self.log = [0] * self.q
        x = 1
        for i in range(self.order):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.q:
                x ^= prim
        for i in range(self.order, 2 * self.order):
            self.exp[i] = self.exp[i - self.order]

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^m)")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b]) % self.order]

    def inv(self, a: int) -> int:
        return self.div(1, a)

    def pow_alpha(self, e: int) -> int:
        return self.exp[e % self.order]

    # polynomials: list of coefficients, index = power of x (low -> high)
    def poly_eval(self, p: list[int], x: int) -> int:
        y = 0
        for coef in reversed(p):
            y = self.mul(y, x) ^ coef
        return y

    def poly_mul(self, p: list[int], q: list[int]) -> list[int]:
        r = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            if a == 0:
                continue
            for j, b in enumerate(q):
                r[i + j] ^= self.mul(a, b)
        return r


# --------------------------------------------------------------------------
# Reed-Solomon code (systematic, first consecutive root alpha^0)
# --------------------------------------------------------------------------


class ReedSolomon:
    def __init__(self, n: int, k: int, m: int = 7):
        self.gf = GF2m(m)
        if not (0 < k < n <= self.gf.order):
            raise ValueError(f"invalid RS parameters n={n}, k={k}, m={m}")
        self.n, self.k = n, k
        self.nsym = n - k
        self.t = self.nsym // 2
        g = [1]
        for i in range(self.nsym):
            g = self.gf.poly_mul(g, [self.gf.pow_alpha(i), 1])  # (x + a^i)
        self.gen = g

    def encode(self, msg: list[int]) -> list[int]:
        """msg: k symbols -> codeword: n symbols = [parity (n-k) | msg (k)]."""
        if len(msg) != self.k:
            raise ValueError("message length must be k")
        gf = self.gf
        # remainder of msg(x) * x^(n-k) divided by g(x)  (LFSR division)
        rem = [0] * self.nsym
        for sym in reversed(msg):  # highest power first
            feedback = sym ^ rem[-1]
            for j in range(self.nsym - 1, 0, -1):
                rem[j] = rem[j - 1] ^ gf.mul(feedback, self.gen[j])
            rem[0] = gf.mul(feedback, self.gen[0])
        return rem + list(msg)

    def decode(self, received: list[int]) -> list[int] | None:
        """Returns the k message symbols, or None if decoding fails."""
        gf = self.gf
        r = list(received)
        synd = [gf.poly_eval(r, gf.pow_alpha(i)) for i in range(self.nsym)]
        if not any(synd):
            return r[self.nsym:]
        # Berlekamp-Massey -> error locator Lambda(x)
        C, B = [1], [1]
        L, shift, b = 0, 1, 1
        for step in range(self.nsym):
            d = synd[step]
            for i in range(1, L + 1):
                if i < len(C):
                    d ^= gf.mul(C[i], synd[step - i])
            if d == 0:
                shift += 1
                continue
            coef = gf.div(d, b)
            Bs = [0] * shift + [gf.mul(coef, x) for x in B]
            T = list(C)
            if len(Bs) > len(C):
                C = C + [0] * (len(Bs) - len(C))
            for i, x in enumerate(Bs):
                C[i] ^= x
            if 2 * L <= step:
                L = step + 1 - L
                B, b, shift = T, d, 1
            else:
                shift += 1
        lam = C[: L + 1]
        if L > self.t:
            return None
        # Chien search: error at position j  <=>  Lambda(alpha^-j) == 0
        positions = [j for j in range(self.n) if gf.poly_eval(lam, gf.pow_alpha(-j)) == 0]
        if len(positions) != L:
            return None
        # Forney: Omega = S(x) * Lambda(x) mod x^(2t)
        omega = gf.poly_mul(synd, lam)[: self.nsym]
        dlam = [lam[i] if i % 2 == 1 else 0 for i in range(1, len(lam))]  # formal derivative
        for j in positions:
            xj = gf.pow_alpha(j)
            xinv = gf.pow_alpha(-j)
            num = gf.mul(xj, gf.poly_eval(omega, xinv))
            den = gf.poly_eval(dlam, xinv)
            if den == 0:
                return None
            r[j] ^= gf.div(num, den)
        if any(gf.poly_eval(r, gf.pow_alpha(i)) for i in range(self.nsym)):
            return None
        return r[self.nsym:]


# --------------------------------------------------------------------------
# Augmented Hadamard code RM(1, r): (r+1) bits -> 2^r bits
# --------------------------------------------------------------------------


class Hadamard:
    def __init__(self, r: int = 6):
        self.r = r
        self.length = 1 << r          # 64
        self.info_bits = r + 1        # 7
        idx = np.arange(self.length)
        # bit matrix: G[i, x] = i-th bit of x
        self._x_bits = ((idx[:, None] >> np.arange(r)[None, :]) & 1).astype(np.uint8)  # (64, r)

    def encode_symbol(self, sym: int) -> np.ndarray:
        a0 = sym & 1
        a = np.array([(sym >> (i + 1)) & 1 for i in range(self.r)], dtype=np.uint8)
        return ((self._x_bits @ a) % 2 ^ a0).astype(np.uint8)

    def encode(self, symbols: list[int]) -> np.ndarray:
        return np.concatenate([self.encode_symbol(s) for s in symbols])

    @staticmethod
    def _fwht(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.int64).copy()
        h = 1
        n = v.shape[-1]
        while h < n:
            v = v.reshape(*v.shape[:-1], n // (2 * h), 2, h)
            a, b = v[..., 0, :].copy(), v[..., 1, :].copy()
            v[..., 0, :], v[..., 1, :] = a + b, a - b
            v = v.reshape(*v.shape[:-3], n)
            h *= 2
        return v

    def decode(self, bits: np.ndarray) -> list[int]:
        """Maximum-likelihood decoding of all 64-bit blocks via the fast Walsh-Hadamard transform."""
        blocks = bits.reshape(-1, self.length)
        signs = 1 - 2 * blocks.astype(np.int64)        # 0 -> +1, 1 -> -1
        spec = self._fwht(signs)                       # (nblocks, 64)
        best = np.argmax(np.abs(spec), axis=1)
        a0 = (spec[np.arange(len(best)), best] < 0).astype(int)
        # best is the integer whose bits are (a1..ar)
        return [int(a0_i) | (int(b) << 1) for a0_i, b in zip(a0, best)]


if __name__ == "__main__":  # quick self-test
    rng = np.random.default_rng(0)
    rs = ReedSolomon(48, 20)
    for trial in range(200):
        msg = list(rng.integers(0, 128, 20))
        cw = rs.encode(msg)
        nerr = rng.integers(0, rs.t + 1)
        pos = rng.choice(48, nerr, replace=False)
        for p in pos:
            cw[p] ^= int(rng.integers(1, 128))
        assert rs.decode(cw) == msg, trial
    had = Hadamard()
    for trial in range(200):
        syms = list(rng.integers(0, 128, 10))
        cw = had.encode(syms)
        for blk in range(10):
            flip = rng.choice(64, 15, replace=False)
            cw[blk * 64 + flip] ^= 1
        assert had.decode(cw) == syms
    print("ecc_codes self-test OK: RS(48,20) corrects", rs.t, "symbols; Hadamard corrects 15/64 bits")
