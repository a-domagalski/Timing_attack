"""
Brumley-Boneh style CRT-RSA timing-attack victim.

Role
----
This is the victim for the remote-timing attack of

    D. Brumley & D. Boneh, "Remote Timing Attacks are Practical",
    USENIX Security 2003,

which itself builds on

    P. L. Montgomery, Math. Comp. 1985      (the extra reduction),
    W. Schindler, CHES 2000                 (extra reductions leak `g mod q`),
    J.-F. Dhem et al., CARDIS 1998          (practical timing attack).

Unlike the square-and-multiply victims elsewhere in this bundle (main.py,
attack_refined/victim_cext.py), this victim performs a full **CRT** RSA
decryption:

    m = c^d mod N,   N = p*q  (q < p),

computed as  m1 = c^{d1} mod p,  m2 = c^{d2} mod q,  recombined with the CRT.
Both half-exponentiations use the *unprotected* C Montgomery core
(attack_refined/cext/montmul.dll via montmul_c), whose conditional final
subtraction ("extra reduction") is the leak.

Why CRT matters (the whole point of the attack)
-----------------------------------------------
Kocher's original attack targets the exponent d; CRT decryption never
exponentiates by d mod N, so it is immune to Kocher.  But CRT reduces the
ciphertext modulo the secret prime q, so the decryption time depends on the
secret factor.  The attacker therefore recovers q directly (a binary search for
q), and once N is factored the private key falls out.

Schindler's channel and the R^{-1} trick
-----------------------------------------
The expected number of extra reductions during g^{d_q} mod q is (Schindler,
eq. 1 in Brumley-Boneh) proportional to (g mod q)/(2R): it rises as g approaches
a multiple of q from below and drops sharply just above.  A raw ciphertext c is
put into Montgomery form as c*R mod q, which *scrambles* the magnitude, so the
attacker (see attack_brumley_boneh.py) submits c = g*R^{-1} mod N.  The victim's
Montgomery conversion then yields exactly (g mod q) as the base, preserving the
magnitude relationship that carries the signal.  (R = 2^(64*limbs) and
R^{-1} mod N are public.)  This victim is a *plain* CRT-RSA decryptor; the trick
lives entirely on the attacker side, exactly as in the paper's Step 2.

Timing modes (same convention as main.py / victim_cext.py)
----------------------------------------------------------
    exact : report the exact number of Montgomery extra reductions performed in
            the whole CRT decryption (ex_p + ex_q) - the idealised, noise-free
            oracle used to demonstrate recovery deterministically (the software
            analogue of Dhem's fixed-clock smartcard cycle counts).
    real  : report the decryption time in SECONDS, from the minimum __rdtsc
            cycle count (timed INSIDE C) over --batch ladder evaluations,
            divided by the calibrated TSC frequency - the genuine, noisy
            wall-clock channel.

I/O protocol (subprocess mode; must stay in sync with the attacker)
-------------------------------------------------------------------
    <- n  (hex)
    <- e  (hex)
    loop:
        -> c            (hex ciphertext, one per line)
        <- observed     (float: extra-reduction count | seconds)
        <- m            (hex plaintext = c^d mod N)

The attacker normally imports :class:`CRTRSAVictim` and drives it in-process
(the Brumley-Boneh attack needs ~10^5-10^6 queries; a subprocess pipe would
dominate the cost and, in real mode, the measured region).  The subprocess CLI
is provided for protocol parity with the other victims in this bundle.
"""

import argparse
import json
import os
import sys
from math import gcd

# --- reuse the C Montgomery core from the sibling attack_refined/ ----------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REFINED = os.path.join(_ROOT, "attack_refined")
for _p in (_REFINED, _ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import montmul_c as mc  # noqa: E402
from sympy import randprime  # noqa: E402


# --------------------------------------------------------------------------- #
# Key generation (fast demo key; balanced primes with q < p so we target q).
# --------------------------------------------------------------------------- #
def generate_crt_key(bits: int, e: int = 65537) -> dict:
    """Return a CRT-RSA key dict for a random ~`bits` modulus with q < p."""
    half = bits // 2
    while True:
        p = randprime(1 << (half - 1), 1 << half)
        q = randprime(1 << (half - 1), 1 << half)
        if p == q:
            continue
        if q > p:  # convention: q is the SMALLER factor (the attack target)
            p, q = q, p
        n = p * q
        if n.bit_length() != bits:
            continue
        phi = (p - 1) * (q - 1)
        if gcd(e, phi) != 1:
            continue
        d = pow(e, -1, phi)
        return {
            "bits": bits,
            "n": n,
            "e": e,
            "d": d,
            "p": p,
            "q": q,
            "d1": d % (p - 1),
            "d2": d % (q - 1),
            "qinv": pow(q, -1, p),
        }


def save_crt_key(path: str, key: dict) -> None:
    with open(path, "w") as fh:
        json.dump({k: str(v) for k, v in key.items()}, fh)


def load_crt_key(path: str) -> dict:
    with open(path) as fh:
        raw = json.load(fh)
    return {k: int(v) for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# The victim.
# --------------------------------------------------------------------------- #
class CRTRSAVictim:
    """Unprotected CRT-RSA decryptor over the C Montgomery core.

    Parameters
    ----------
    key : dict from :func:`generate_crt_key` / :func:`load_crt_key`.
    timing : "exact" | "real".
    batch : real mode - __rdtsc-timed ladder evaluations averaged per half.
    repeat : real mode - keep the minimum over this many batch-averages.
    """

    def __init__(self, key: dict, timing: str = "exact", batch: int = 64,
                 repeat: int = 1):
        self.n = key["n"]
        self.e = key["e"]
        self.d = key["d"]
        self.p = key["p"]
        self.q = key["q"]
        self.d1 = key["d1"]
        self.d2 = key["d2"]
        self.qinv = key["qinv"]
        self.timing = timing
        self.batch = max(1, int(batch))
        self.repeat = max(1, int(repeat))
        # queries = logical decryption submissions; ladders = C exponentiations.
        self.queries = 0
        self.hz = mc.tsc_hz(0.1) if timing == "real" else None

    # -- Montgomery radix for the q-exponentiation (public; used by attacker) - #
    def q_radix(self):
        s = (self.q.bit_length() + 63) // 64
        return 1 << (64 * s)

    def decrypt_instrumented(self, cipher: int, do_time: bool = False) -> dict:
        """One full CRT decryption; returns plaintext + per-half diagnostics."""
        g = cipher % self.n
        m1, ex_p, cyc_p = mc.modexp(g, self.d1, self.p,
                                    do_time=do_time, batch=self.batch)
        m2, ex_q, cyc_q = mc.modexp(g, self.d2, self.q,
                                    do_time=do_time, batch=self.batch)
        h = (self.qinv * (m1 - m2)) % self.p
        m = m2 + h * self.q
        self.queries += 1
        return {
            "m": m, "ex_p": ex_p, "ex_q": ex_q, "ex_total": ex_p + ex_q,
            "cyc_p": cyc_p, "cyc_q": cyc_q,
        }

    def measure(self, cipher: int) -> float:
        """Return the observed side-channel value for one decryption of
        `cipher`: extra-reduction count (exact) or seconds (real)."""
        if self.timing == "real":
            best = None
            for _ in range(self.repeat):
                d = self.decrypt_instrumented(cipher, do_time=True)
                avg_cycles = d["cyc_p"] / self.batch + d["cyc_q"] / self.batch
                best = avg_cycles if best is None else min(best, avg_cycles)
            return best / self.hz
        d = self.decrypt_instrumented(cipher, do_time=False)
        return float(d["ex_total"])


# --------------------------------------------------------------------------- #
# Self-test: correctness of CRT decryption + determinism of the leak.
# --------------------------------------------------------------------------- #
def self_test(bits: int = 256, trials: int = 200) -> bool:
    import random

    key = generate_crt_key(bits)
    vic = CRTRSAVictim(key, timing="exact")
    n, d = key["n"], key["d"]
    ok = True
    for _ in range(trials):
        g = random.randrange(2, n)
        info = vic.decrypt_instrumented(g)
        if info["m"] != pow(g, d, n):
            ok = False
            break
    # determinism of the extra-reduction count
    g = random.randrange(2, n)
    a = vic.decrypt_instrumented(g)["ex_total"]
    b = vic.decrypt_instrumented(g)["ex_total"]
    det = a == b
    print("[bb_victim self-test] bits=%d  CRT-correct=%s  deterministic-leak=%s"
          % (bits, ok, det))
    print("  n = %d bits, q < p = %s" % (n.bit_length(), key["q"] < key["p"]))
    return ok and det


# --------------------------------------------------------------------------- #
# Subprocess CLI (protocol parity with the other victims in the bundle).
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Brumley-Boneh CRT-RSA timing victim.")
    ap.add_argument("--key-bits", type=int, default=256)
    ap.add_argument("--timing", choices=["exact", "real"], default="exact")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--reuse-key", action="store_true")
    ap.add_argument("--key-file", default="subprocess_key_bb.json")
    ap.add_argument("--self-test", action="store_true",
                    help="run a correctness self-test and exit")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        ok = self_test(args.key_bits)
        sys.exit(0 if ok else 1)

    key = None
    if args.reuse_key and os.path.exists(args.key_file):
        try:
            key = load_crt_key(args.key_file)
            if key["n"].bit_length() != args.key_bits:
                key = None
        except Exception:
            key = None
    if key is None:
        key = generate_crt_key(args.key_bits)
    try:
        save_crt_key(args.key_file, key)
    except OSError:
        pass

    vic = CRTRSAVictim(key, timing=args.timing, batch=args.batch,
                       repeat=args.repeat)

    print("{0:X}".format(key["n"]), flush=True)
    print("{0:X}".format(key["e"]), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            c = int(line, 16) % key["n"]
        except ValueError:
            continue
        info = vic.decrypt_instrumented(
            c, do_time=(args.timing == "real"))
        if args.timing == "real":
            avg = info["cyc_p"] / vic.batch + info["cyc_q"] / vic.batch
            observed = avg / vic.hz
        else:
            observed = float(info["ex_total"])
        print(repr(observed), flush=True)
        print("{0:X}".format(info["m"]), flush=True)


if __name__ == "__main__":
    main()
