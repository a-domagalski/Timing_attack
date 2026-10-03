#!/usr/bin/env python3
"""
Coppersmith completion for the Brumley-Boneh attack -- WSL / fpylll finisher.

WHY THIS EXISTS
---------------
The in-repo pure-Python completion (``coppersmith.py``) uses sympy's integer LLL.
It works on easy cases but is numerically fragile near the N**(1/4) edge and, in
practice, FAILED to factor a specific provable 256-bit key even given the true
top 88 bits of q (while it factored random keys at the same settings). That is a
weakness of the LLL implementation, not of the recovered bits.

This script does the SAME Howgrave-Graham lattice construction but reduces it
with **fpylll** (the FPLLL C++ library), which is fast and robust and reaches
close to the theoretical N**(1/4) bound. fpylll does not build on this project's
Windows/Py3.13 venv (no wheel; needs GMP/MPFR/QD + a C++ toolchain), so the
completion runs in **WSL/Ubuntu** while the multi-hour timing recovery stays on
Windows. The two halves exchange a tiny JSON file:

    Windows attack_remote.py (full mode)  --writes-->  completion_input.json
    WSL   coppersmith_fpylll.py           --reads--->  completion_output.json

No secret is in the handoff: only N, e, the recovered high-bits PREFIX of q, and
how many low bits are unknown.

USAGE (inside the WSL venv where `import fpylll` works)
-------------------------------------------------------
    source ~/cop/bin/activate
    python3 coppersmith_fpylll.py /mnt/d/.../remote_<ts>_kb256/completion_input.json

    # options:
    #   --brute B   also brute-force the B least-significant KNOWN bits of the
    #               prefix (buys ~B bits of margin when the recovery is right at
    #               the N**(1/4) edge, e.g. only ~64-66 correct top bits).
    #   --dim-cap D max lattice dimension to try (default 40; fpylll handles it).

Reads `completion_input.json`:
    { "n_hex": ..., "e": ..., "prefix_hex": <a, low bits 0>,
      "unknown_bits": <log2 X>, "beta": 0.5 }
Writes `completion_output.json` next to it:
    { "factored": bool, "q_hex": ..., "p_hex": ..., "d_hex": ..., ... }
"""

import argparse
import json
import os
import sys
import time
from math import comb

try:
    from fpylll import IntegerMatrix, LLL
except ImportError:
    sys.exit("fpylll not importable -- run inside the WSL venv where "
             "`python -c 'import fpylll'` succeeds (see project README).")

import sympy


# --------------------------------------------------------------------------- #
# Lattice-parameter selection (same math as coppersmith.py, no small dim cap).
# --------------------------------------------------------------------------- #
def _achievable_bits(N_bits, beta, m, t):
    w = m + 1 + t
    Sx = m * (m + 1) // 2 + t * m + t * (t + 1) // 2
    if Sx == 0:
        return 0.0
    exp_of_N = (beta * m * w - m * (m + 1) / 2.0) / Sx
    return exp_of_N * N_bits


def choose_mt(N_bits, unknown_bits, beta=0.5, dim_cap=40, margin=1.0):
    """Smallest-dimension (m, t) whose Howgrave-Graham bound covers
    `unknown_bits`. dim_cap is generous (fpylll reduces large bases fine)."""
    best = None
    for m in range(1, dim_cap):
        for t in range(0, 2 * m + 1):
            if m + 1 + t > dim_cap:
                break
            if _achievable_bits(N_bits, beta, m, t) >= unknown_bits + margin:
                dim = m + 1 + t
                if best is None or dim < best[0]:
                    best = (dim, m, t)
        if best is not None and best[0] <= m + 1:
            break
    return (best[1], best[2]) if best else None


# --------------------------------------------------------------------------- #
# Coppersmith via fpylll.
# --------------------------------------------------------------------------- #
def factor_with_high_bits(N, a, X, beta=0.5, m=None, t=None, dim_cap=40):
    """Return a nontrivial factor q of N with a <= q < a + X, or None.

    Same Howgrave-Graham lattice as coppersmith.py; fpylll LLL for the reduction.
    """
    if a <= 1 or X <= 1:
        return a if (a > 1 and N % a == 0) else None
    if N % a == 0 and a > 1:
        return a

    N_bits = N.bit_length()
    unknown_bits = X.bit_length()
    if m is None or t is None:
        mt = choose_mt(N_bits, unknown_bits, beta=beta, dim_cap=dim_cap)
        if mt is None:
            return None
        m, t = mt
    dim = m + 1 + t

    def fpow(i):                     # (x + a)^i as coeff list, low power first
        return [comb(i, k) * pow(a, i - k) for k in range(i + 1)]

    # Build integer polynomial rows (same as coppersmith.py).
    polys = []
    for i in range(m + 1):           # N^{m-i} * f(x)^i
        c = fpow(i)
        row = [0] * dim
        Nfac = pow(N, m - i)
        for k, ck in enumerate(c):
            row[k] = ck * Nfac
        polys.append(row)
    fm = fpow(m)
    for j in range(1, t + 1):        # x^j * f(x)^m
        row = [0] * dim
        for k, ck in enumerate(fm):
            row[k + j] = ck
        polys.append(row)

    # Scale column col by X**col so lattice norm matches |h(xX)|.
    Xpow = [pow(X, col) for col in range(dim)]
    scaled = [[row[col] * Xpow[col] for col in range(dim)] for row in polys]

    # fpylll LLL over exact integers.
    A = IntegerMatrix.from_matrix([[int(v) for v in row] for row in scaled])
    LLL.reduction(A)
    reduced = [[A[r, c] for c in range(dim)] for r in range(dim)]

    x = sympy.Symbol("x")
    for vec in reduced:
        coeffs = [vec[col] // Xpow[col] for col in range(dim)]  # exact unscale
        if not any(coeffs):
            continue
        expr = sum(coeffs[col] * x ** col for col in range(dim))
        try:
            roots = sympy.Poly(expr, x).ground_roots()
        except Exception:
            continue
        for r in roots:
            if r.is_integer:
                q = a + int(r)
                if q > 1 and N % q == 0:
                    return q
    return None


def complete(N, prefix, unknown_bits, beta=0.5, brute=0, dim_cap=40):
    """Try Coppersmith; optionally brute-force the `brute` least-significant
    KNOWN bits of the prefix to buy margin at the N**(1/4) edge.

    Brute rationale: if the recovery is right at the limit (say only ~64-66
    correct top bits), widen the unknown window by `brute` bits and try each of
    the 2**brute settings of those bits -- one of them matches the true q, and
    each call now needs `brute` fewer known bits."""
    q = factor_with_high_bits(N, prefix, 1 << unknown_bits, beta=beta,
                              dim_cap=dim_cap)
    if q is not None:
        return q, 0
    if brute <= 0:
        return None, 0
    U = unknown_bits + brute
    base = (prefix >> U) << U         # bits strictly above the widened window
    tried = 0
    for hi in range(1 << brute):
        a = base | (hi << unknown_bits)
        q = factor_with_high_bits(N, a, 1 << U, beta=beta, dim_cap=dim_cap)
        tried += 1
        if q is not None:
            return q, tried
    return None, tried


def complete_sweep(N, prefix, unknown_bits, beta=0.5, brute=0, dim_cap=40,
                   sweep_max=0):
    """Try completion while progressively DISTRUSTING the lowest recovered bits.

    The timing recovery produces a prefix down to some low bit, but a single
    mid-bit error corrupts everything below it while the TOP bits remain correct.
    Coppersmith only needs the correct top ~half, so we sweep the trusted depth:
    for u = unknown_bits, unknown_bits+1, ..., sweep_max, zero the low u bits of
    the prefix (treat them as unknown) and try to factor. The first u whose
    trusted top (N.bits/2 - u bits) is entirely correct AND within fpylll's reach
    factors. Returns (q, u_used, brute_tried) or (None, None, tried)."""
    hi = max(unknown_bits, sweep_max)
    for u in range(unknown_bits, hi + 1):
        a = (prefix >> u) << u
        q, tried = complete(N, a, u, beta=beta, brute=brute, dim_cap=dim_cap)
        if q is not None and N % q == 0:
            return q, u, tried
    return None, None, 0


def main():
    ap = argparse.ArgumentParser(description="fpylll Coppersmith finisher.")
    ap.add_argument("input", help="path to completion_input.json")
    ap.add_argument("--brute", type=int, default=0,
                    help="brute-force this many low KNOWN bits for edge margin")
    ap.add_argument("--sweep-max", type=int, default=0,
                    help="if >unknown_bits, sweep the trusted depth up to this "
                         "many unknown bits (tolerates mid-bit recovery errors "
                         "below a correct top prefix)")
    ap.add_argument("--dim-cap", type=int, default=40)
    ap.add_argument("--true-q-hex", default=None,
                    help="optional: verify the recovered q matches this")
    args = ap.parse_args()

    with open(args.input) as fh:
        cfg = json.load(fh)
    N = int(cfg["n_hex"], 16)
    e = int(cfg["e"])
    beta = float(cfg.get("beta", 0.5))

    # Candidate prefixes: prefer the full beam-leaf list (the fully-correct
    # prefix is often NOT the top-scoring leaf on the noisy oracle); fall back to
    # the single recovered prefix for older/handwritten inputs.
    cands = cfg.get("candidates")
    if not cands:
        cands = [{"prefix_hex": cfg["prefix_hex"],
                  "unknown_bits": int(cfg["unknown_bits"])}]

    print("[fpylll-coppersmith] N=%d bits | %d candidate prefix(es) | beta=%.3f "
          "| brute=%d | sweep_max=%d"
          % (N.bit_length(), len(cands), beta, args.brute, args.sweep_max))

    t0 = time.time()
    q = None
    u_used = None
    tried = 0
    hit = None
    # Two passes: first try EVERY candidate with no brute (a fully-correct
    # prefix factors here in ~0.1-0.4s), only then spend the slow brute margin.
    brute_levels = [0] + ([args.brute] if args.brute and args.brute > 0 else [])
    for brute in brute_levels:
        for ci, cand in enumerate(cands):
            prefix = int(cand["prefix_hex"], 16)
            unknown_bits = int(cand["unknown_bits"])
            lead = cand.get("leading_correct")
            if args.sweep_max and args.sweep_max > unknown_bits:
                qi, u_i, ti = complete_sweep(N, prefix, unknown_bits, beta=beta,
                                             brute=brute, dim_cap=args.dim_cap,
                                             sweep_max=args.sweep_max)
            else:
                qi, ti = complete(N, prefix, unknown_bits, beta=beta,
                                  brute=brute, dim_cap=args.dim_cap)
                u_i = unknown_bits if qi is not None else None
            tried += ti
            if qi is not None and N % qi == 0:
                q, u_used, hit = qi, u_i, (ci, lead)
                print("  candidate %d/%d (leading_correct=%s) FACTORED "
                      "(unknown_bits used=%s, brute=%d)"
                      % (ci + 1, len(cands), lead, u_i, brute))
                break
        if q is not None:
            break
    dt = time.time() - t0

    out = {"factored": False, "seconds": dt, "brute_tried": tried,
           "unknown_bits_used": u_used, "candidates_tried": len(cands),
           "winning_candidate": (hit[0] if hit else None),
           "winning_leading_correct": (hit[1] if hit else None),
           "input": os.path.abspath(args.input)}
    if q is not None and N % q == 0:
        p = N // q
        try:
            d = pow(e, -1, (p - 1) * (q - 1))
        except Exception:
            d = None
        out.update({"factored": True, "q_hex": "%X" % q, "p_hex": "%X" % p,
                    "d_hex": ("%X" % d) if d is not None else None})
        print("  >>> FACTORED in %.2fs (unknown_bits used=%s, brute tried=%d)"
              % (dt, u_used, tried))
        print("      q = %X" % q)
        if args.true_q_hex:
            match = (q == int(args.true_q_hex, 16)
                     or p == int(args.true_q_hex, 16))
            out["matches_true_q"] = match
            print("      matches true q: %s" % match)
    else:
        print("  >>> NOT factored (%.2fs, brute tried=%d). Need more correct "
              "top bits, or try a larger --brute/--sweep-max." % (dt, tried))

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                            "completion_output.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("  output: %s" % out_path)
    sys.exit(0 if out["factored"] else 2)


if __name__ == "__main__":
    main()
