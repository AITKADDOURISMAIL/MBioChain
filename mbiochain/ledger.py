"""
Minimal permissioned blockchain (Proof-of-Authority) with an on-chain identity registry.

Registry state per DID:
    pk          current public key (compressed SEC1, 33 bytes)
    helper_hash SHA-256 of the off-chain helper data
    epoch       key epoch (incremented at each rotation / recovery)
    guardians   list of guardian public keys, threshold t
    revoked     list of revoked public keys

Transactions (all ECDSA/secp256k1 signed):
    REGISTER   signed by the new key
    AUTH       challenge-response proof of possession (not stored, verified off-chain or on-chain)
    ROTATE     signed by the *current* key  -> voluntary key change (new epoch)
    RECOVER    signed by >= t guardians     -> key lost or biometric compromised
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


def pk_bytes(pk: ec.EllipticCurvePublicKey) -> bytes:
    return pk.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint)


def pk_from_bytes(b: bytes) -> ec.EllipticCurvePublicKey:
    return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), b)


def address(pk: ec.EllipticCurvePublicKey) -> str:
    """Illustrative address: last 20 bytes of SHA3-256 of the uncompressed point."""
    raw = pk.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return "0x" + hashlib.sha3_256(raw[1:]).digest()[-20:].hex()


def sign(sk: ec.EllipticCurvePrivateKey, msg: bytes) -> bytes:
    return sk.sign(msg, ec.ECDSA(hashes.SHA256()))


def verify(pk: ec.EllipticCurvePublicKey, sig: bytes, msg: bytes) -> bool:
    try:
        pk.verify(sig, msg, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


@dataclass
class Tx:
    kind: str
    body: dict                    # hex-encoded fields
    sigs: list[str]               # hex signatures

    def message(self) -> bytes:
        return canonical({"kind": self.kind, "body": self.body})

    def to_dict(self) -> dict:
        return {"kind": self.kind, "body": self.body, "sigs": self.sigs}


@dataclass
class Block:
    index: int
    prev_hash: str
    timestamp: float
    txs: list[dict]
    merkle_root: str
    authority_sig: str = ""
    hash: str = ""

    def header(self) -> bytes:
        return canonical({"i": self.index, "prev": self.prev_hash, "ts": self.timestamp,
                          "root": self.merkle_root})


def merkle_root(items: list[bytes]) -> str:
    if not items:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(x).digest() for x in items]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


@dataclass
class IdentityRecord:
    pk: str
    helper_hash: str
    epoch: int
    guardians: list[str]
    threshold: int
    nonce: int = 0
    revoked: list[str] = field(default_factory=list)


class Ledger:
    def __init__(self, authority_sk: ec.EllipticCurvePrivateKey | None = None):
        self.authority_sk = authority_sk or ec.generate_private_key(ec.SECP256K1())
        self.chain: list[Block] = []
        self.mempool: list[Tx] = []
        self.registry: dict[str, IdentityRecord] = {}
        self._append_block([])  # genesis

    # ----------------------------------------------------------------- blocks
    def _append_block(self, txs: list[Tx]) -> Block:
        prev = self.chain[-1].hash if self.chain else "0" * 64
        blk = Block(len(self.chain), prev, time.time(), [t.to_dict() for t in txs],
                    merkle_root([canonical(t.to_dict()) for t in txs]))
        blk.hash = hashlib.sha256(blk.header()).hexdigest()
        blk.authority_sig = sign(self.authority_sk, blk.header()).hex()
        self.chain.append(blk)
        return blk

    def submit(self, tx: Tx) -> bool:
        ok, _ = self._validate(tx, dry_run=True)
        if ok:
            self.mempool.append(tx)
        return ok

    def seal(self) -> Block:
        """Validate and apply mempool transactions, then append a block."""
        applied = []
        for tx in self.mempool:
            ok, _ = self._validate(tx, dry_run=False)
            if ok:
                applied.append(tx)
        self.mempool = []
        return self._append_block(applied)

    def verify_chain(self) -> bool:
        apk = self.authority_sk.public_key()
        for i, b in enumerate(self.chain):
            if b.hash != hashlib.sha256(b.header()).hexdigest():
                return False
            if i and b.prev_hash != self.chain[i - 1].hash:
                return False
            if b.merkle_root != merkle_root([canonical(t) for t in b.txs]):
                return False
            if not verify(apk, bytes.fromhex(b.authority_sig), b.header()):
                return False
        return True

    # --------------------------------------------------------------- contract
    def _validate(self, tx: Tx, dry_run: bool) -> tuple[bool, str]:
        b, msg = tx.body, tx.message()
        did = b.get("did", "")
        rec = self.registry.get(did)

        if tx.kind == "REGISTER":
            if rec is not None:
                return False, "DID exists"
            pk = pk_from_bytes(bytes.fromhex(b["pk"]))
            if not verify(pk, bytes.fromhex(tx.sigs[0]), msg):
                return False, "bad signature"
            if not dry_run:
                self.registry[did] = IdentityRecord(b["pk"], b["helper_hash"], b["epoch"],
                                                    b["guardians"], b["threshold"])
            return True, "ok"

        if rec is None:
            return False, "unknown DID"
        if b.get("nonce") != rec.nonce:
            return False, "bad nonce (replay)"

        if tx.kind == "ROTATE":
            if not verify(pk_from_bytes(bytes.fromhex(rec.pk)), bytes.fromhex(tx.sigs[0]), msg):
                return False, "not signed by current key"
        elif tx.kind == "RECOVER":
            signers = set()
            for g in rec.guardians:
                gpk = pk_from_bytes(bytes.fromhex(g))
                if any(verify(gpk, bytes.fromhex(s), msg) for s in tx.sigs):
                    signers.add(g)
            if len(signers) < rec.threshold:
                return False, f"only {len(signers)}/{rec.threshold} guardian signatures"
        else:
            return False, "unknown tx kind"

        if b["epoch"] <= rec.epoch:
            return False, "epoch must increase"
        if not dry_run:
            rec.revoked.append(rec.pk)
            rec.pk, rec.helper_hash, rec.epoch = b["new_pk"], b["helper_hash"], b["epoch"]
            rec.nonce += 1
        return True, "ok"

    # ------------------------------------------------------------- queries
    def current_pk(self, did: str) -> ec.EllipticCurvePublicKey | None:
        rec = self.registry.get(did)
        return pk_from_bytes(bytes.fromhex(rec.pk)) if rec else None

    def is_revoked(self, did: str, pk: ec.EllipticCurvePublicKey) -> bool:
        rec = self.registry.get(did)
        return bool(rec) and pk_bytes(pk).hex() in rec.revoked


# --------------------------------------------------------------------- tx builders


def make_register(did, sk, helper_hash: bytes, epoch, guardians, threshold) -> Tx:
    body = {"did": did, "pk": pk_bytes(sk.public_key()).hex(), "helper_hash": helper_hash.hex(),
            "epoch": epoch, "threshold": threshold,
            "guardians": [pk_bytes(g.public_key() if hasattr(g, "private_numbers") else g).hex()
                          for g in guardians]}
    tx = Tx("REGISTER", body, [])
    tx.sigs = [sign(sk, tx.message()).hex()]
    return tx


def make_rotate(did, old_sk, new_pk, helper_hash: bytes, epoch, nonce) -> Tx:
    body = {"did": did, "new_pk": pk_bytes(new_pk).hex(), "helper_hash": helper_hash.hex(),
            "epoch": epoch, "nonce": nonce}
    tx = Tx("ROTATE", body, [])
    tx.sigs = [sign(old_sk, tx.message()).hex()]
    return tx


def make_recover(did, guardian_sks, new_pk, helper_hash: bytes, epoch, nonce) -> Tx:
    body = {"did": did, "new_pk": pk_bytes(new_pk).hex(), "helper_hash": helper_hash.hex(),
            "epoch": epoch, "nonce": nonce}
    tx = Tx("RECOVER", body, [])
    tx.sigs = [sign(g, tx.message()).hex() for g in guardian_sks]
    return tx


def challenge_response(sk, challenge: bytes) -> bytes:
    return sign(sk, b"MBioChain|auth|" + challenge)


def check_response(pk, challenge: bytes, sig: bytes) -> bool:
    return verify(pk, sig, b"MBioChain|auth|" + challenge)
