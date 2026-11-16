"""
End-to-end MBioChain scenario (synthetic templates):
  1. enrollment (iris + fingerprint + PIN) -> secp256k1 key, helper data
  2. REGISTER on the blockchain (DID, pk, hash(helper), 2-of-3 guardians)
  3. authentication by challenge-response from a NEW biometric capture
  4. failures: wrong PIN, impostor biometrics
  5. ROTATE (new epoch: new key + new helper data, old key revoked)
  6. RECOVER with 2-of-3 guardians (e.g. stolen device + compromised biometrics)
"""
import os

from mbiochain import ledger as L
from mbiochain.bio_key import Params, enroll, recover
from mbiochain.data import SyntheticDB


def main():
    db = SyntheticDB(n_subjects=3, n_samples=4, seed=7)
    alice, mallory = 0, 1
    p = Params()
    chain = L.Ledger()
    guardians = [L.ec.generate_private_key(L.ec.SECP256K1()) for _ in range(3)]
    did = "did:mbio:alice"

    print("1) Enrollment")
    sk, pk, helper = enroll(db.iris[alice][0], db.finger[alice][0], "4821", p, epoch=0)
    print("   address      :", L.address(pk))
    print("   helper data  :", helper.size_bytes(), "bytes (stored off-chain), hash on-chain")

    print("2) REGISTER")
    assert chain.submit(L.make_register(did, sk, helper.digest(), 0, guardians, threshold=2))
    blk = chain.seal()
    print(f"   block #{blk.index} {blk.hash[:16]}...  registry pk = {chain.registry[did].pk[:20]}...")

    print("3) Authentication with a fresh capture")
    r = recover(db.iris[alice][2], db.finger[alice][2], "4821", helper)
    assert r is not None, "genuine recovery failed"
    sk2, pk2, shift = r
    chal = os.urandom(32)
    ok = L.check_response(chain.current_pk(did), chal, L.challenge_response(sk2, chal))
    print(f"   key recovered (iris shift {shift:+d}), signature accepted by chain: {ok}")

    print("4) Attacks")
    print("   wrong PIN          ->", "REJECTED" if recover(db.iris[alice][2], db.finger[alice][2], "0000", helper) is None else "ACCEPTED")
    print("   impostor biometrics->", "REJECTED" if recover(db.iris[mallory][0], db.finger[mallory][0], "4821", helper) is None else "ACCEPTED")

    print("5) ROTATE (voluntary re-enrollment, epoch 1)")
    sk_new, pk_new, helper_new = enroll(db.iris[alice][1], db.finger[alice][1], "4821", p, epoch=1)
    assert chain.submit(L.make_rotate(did, sk2, pk_new, helper_new.digest(), 1, nonce=0))
    chain.seal()
    print("   old key revoked:", chain.is_revoked(did, pk), "| new key active:",
          chain.current_pk(did).public_numbers() == pk_new.public_numbers())
    chal = os.urandom(32)
    print("   old key now authenticates:", L.check_response(chain.current_pk(did), chal, L.challenge_response(sk2, chal)))

    print("6) RECOVER with 2-of-3 guardians (epoch 2)")
    sk_r, pk_r, helper_r = enroll(db.iris[alice][3], db.finger[alice][3], "7310", p, epoch=2)
    bad = L.make_recover(did, guardians[:1], pk_r, helper_r.digest(), 2, nonce=1)
    print("   with 1 guardian  ->", "accepted" if chain.submit(bad) else "rejected")
    good = L.make_recover(did, guardians[:2], pk_r, helper_r.digest(), 2, nonce=1)
    print("   with 2 guardians ->", "accepted" if chain.submit(good) else "rejected")
    chain.seal()
    print("   epoch on chain:", chain.registry[did].epoch, "| revoked keys:", len(chain.registry[did].revoked))
    print("Chain valid:", chain.verify_chain(), "| blocks:", len(chain.chain))


if __name__ == "__main__":
    main()
