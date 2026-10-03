"""
Attack target for the Montgomery-multiplication RSA timing attack.

Role
----
This process plays the role of the victim.  It:
  1. Generates (or reloads) an RSA key pair.
  2. Prints the public modulus ``n`` and public exponent ``e`` (hex).
  3. On every line read from stdin it interprets the line as a hex ciphertext
     ``c``, performs an *unprotected* square-and-multiply Montgomery
     exponentiation ``c**d mod n`` (the decryption), and reports:
         line 1:  the observed "time" for that decryption
         line 2:  the recovered plaintext m = c**d mod n (hex)

I/O protocol (must stay in sync with attack.py)
-----------------------------------------------
    <- n  (hex)
    <- e  (hex)
    loop:
        -> c            (hex ciphertext, one per line)
        <- time         (float; see --timing below)
        <- m            (hex plaintext = c**d mod n)

Configuration (command-line arguments — NO environment variables)
-----------------------------------------------------------------
    --key-bits N     modulus bit length for the demo key (default 256).
    --timing MODE    "exact" (default) reports the exact number of Montgomery
                     extra reductions performed during the decryption — the
                     idealised, noise-free side channel used to verify recovery
                     deterministically.  "real" reports a wall-clock
                     time.perf_counter() measurement (min over --repeat runs);
                     the genuine but very noisy channel.
    --repeat K       in "real" mode, repeat the decryption K times and report
                     the minimum (reduces scheduler jitter). Default 5.
    --reuse-key      if set and --key-file already holds a key of the right
                     bit length, reload it instead of generating a fresh one.
                     Lets the attacker keep a valid measurement cache across
                     runs (the samples depend on n/d).
    --key-file PATH  where the (n, e, d) is persisted / reloaded
                     (default: subprocess_key.txt in the CWD).

References
----------
The victim deliberately uses an *unprotected* Montgomery square-and-multiply so
that the conditional final subtraction leaks, exactly the target modelled by:
  * P. L. Montgomery, Math. Comp. 1985 (the multiplication + final subtraction);
  * W. Schindler, CHES 2000, and C. D. Walter & S. Thompson, CT-RSA 2001
    (the extra reduction as an exponent-bit oracle);
  * J.-F. Dhem et al., CARDIS 1998, and D. Brumley & D. Boneh, USENIX 2003
    (practical / remote realisations of timing this decryption).
The attacker lives in RSA-Timing-Attack/attack.py.
"""

import argparse
import os
import sys
import time
from math import gcd

from sympy import randprime

# Reuse the *vulnerable* Montgomery primitives from custom_rsa so the target
# and the attacker share exactly the same arithmetic.
from custom_rsa import getOmega, MontMul, limbsNr


# --------------------------------------------------------------------------- #
# Key generation (fast demo key; the timing attack does not care how the key
# was produced, only that decryption uses the unprotected sqam_montgomery)
# --------------------------------------------------------------------------- #
def generate_demo_key(bits: int):
    """Return (n, e, d) for a random RSA key of ~`bits` modulus length."""
    half = bits // 2
    e = 65537
    while True:
        p = randprime(1 << (half - 1), 1 << half)
        q = randprime(1 << (half - 1), 1 << half)
        if p == q:
            continue
        n = p * q
        if n.bit_length() != bits:
            continue
        phi = (p - 1) * (q - 1)
        if gcd(e, phi) != 1:
            continue
        d = pow(e, -1, phi)
        return n, e, d


def save_key(path, n, e, d):
    with open(path, "w") as fh:
        fh.write("n: {}\ne: {}\nd: {}\n".format(n, e, d))


def load_key(path):
    with open(path) as fh:
        s = fh.read()
    n = int(s.split("n:")[1].split("\n")[0].strip())
    e = int(s.split("e:")[1].split("\n")[0].strip())
    d = int(s.split("d:")[1].split("\n")[0].strip())
    return n, e, d


# --------------------------------------------------------------------------- #
# Unprotected square-and-multiply Montgomery exponentiation.
# Identical ladder to custom_rsa.sqam_montgomery, but it also counts the
# number of conditional (extra) reductions so we can expose the exact oracle.
#
# The `extra` flag returned by MontMul is the Montgomery final subtraction
# (Montgomery 1985); its data-dependent occurrence is the side channel that
# Schindler (CHES 2000) / Walter-Thompson (CT-RSA 2001) turn into a bit oracle.
# --------------------------------------------------------------------------- #
def sqam_montgomery_counted(base, exponent, n):
    w = 64
    k = limbsNr(n)
    R = 1 << (w * k)
    omega = getOmega(n)

    base_bar = (base * R) % n     # Montgomery form of the base
    res_bar = R % n               # Montgomery form of 1
    extra_reductions = 0

    for bit in bin(exponent)[2:]:
        res_bar, ex = MontMul(res_bar, res_bar, n, omega)   # square
        extra_reductions += ex
        if bit == "1":
            res_bar, ex = MontMul(res_bar, base_bar, n, omega)  # multiply
            extra_reductions += ex

    result, _ = MontMul(res_bar, 1, n, omega)               # back from Mont domain
    return result, extra_reductions


def decrypt_and_measure(c, d, n, timing_mode, repeat):
    """Return (reported_time, plaintext)."""
    if timing_mode == "real":
        best = None
        plaintext = None
        for _ in range(max(1, repeat)):
            start = time.perf_counter()
            plaintext, _ = sqam_montgomery_counted(c, d, n)
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
        return best, plaintext
    else:  # "exact"
        plaintext, extra = sqam_montgomery_counted(c, d, n)
        return float(extra), plaintext


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Montgomery timing-attack victim.")
    ap.add_argument("--key-bits", type=int, default=256)
    ap.add_argument("--timing", choices=["exact", "real"], default="exact")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--reuse-key", action="store_true")
    ap.add_argument("--key-file", default="subprocess_key.txt")
    return ap.parse_args()


def main():
    args = parse_args()

    n = e = d = None
    if args.reuse_key and os.path.exists(args.key_file):
        try:
            n, e, d = load_key(args.key_file)
            if n.bit_length() != args.key_bits:
                n = e = d = None  # cached key is a different size; regenerate
        except Exception:
            n = e = d = None

    if n is None:
        # Generate quietly (any library chatter would corrupt the protocol).
        real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            n, e, d = generate_demo_key(args.key_bits)
        finally:
            sys.stdout.close()
            sys.stdout = real_stdout

    # Persist the key so the run can be audited / the recovered d checked, and
    # so a later --reuse-key run keeps the attacker's measurement cache valid.
    try:
        save_key(args.key_file, n, e, d)
    except OSError:
        pass

    # Handshake: emit n and e in hex.
    print("{0:X}".format(n), flush=True)
    print("{0:X}".format(e), flush=True)

    # Service decryption queries.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            c = int(line, 16) % n
        except ValueError:
            continue
        reported_time, plaintext = decrypt_and_measure(
            c, d, n, args.timing, args.repeat
        )
        print(repr(reported_time), flush=True)
        print("{0:X}".format(plaintext), flush=True)


if __name__ == "__main__":
    main()
