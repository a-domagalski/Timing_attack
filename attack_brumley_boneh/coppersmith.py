"""
Coppersmith completion for the Brumley-Boneh attack.

Brumley & Boneh, "Remote Timing Attacks are Practical" (USENIX 2003), Section 3,
recover only the *top ~half* of the bits of the factor q by timing and then
finish with **Coppersmith's algorithm** to obtain the full factorization:

    "After recovering the half-most significant bits of q, we can use
     Coppersmith's algorithm [3] to retrieve the complete factorization."

This module implements exactly that finish -- the univariate "factoring with
high bits of a factor known" case of Coppersmith's method (Coppersmith 1996;
Howgrave-Graham's 1997 lattice reformulation):

Given N = p*q and an approximation `a` of q with q = a + x0, 0 <= x0 < X,
build a lattice of polynomials that all vanish modulo q^m at x0, LLL-reduce it,
read off a short vector as an integer polynomial h(x) that has x0 as an *integer*
root (Howgrave-Graham's lemma), solve h over the integers, and test a + x0 | N.
The classical bound is X < N^{beta^2} for a divisor q >= N^beta; for balanced
RSA (beta = 1/2) that is X < N^{1/4}, i.e. roughly the top half of q's bits must
be known -- precisely the Brumley-Boneh regime.

Pure Python (a small exact-arithmetic LLL + sympy only for integer root
finding), so it needs no fpylll / sage.  Intended for the demo key sizes in this
bundle (256-bit N); the LLL dimension is kept small on purpose.
"""

from math import comb

import sympy
from sympy import Matrix, Rational


# --------------------------------------------------------------------------- #
# LLL reduction.
#
# Uses sympy's built-in fraction-free integer LLL (sympy.Matrix.lll, backed by
# the DomainMatrix ZZ implementation) -- orders of magnitude faster than a
# naive Fraction Gram-Schmidt that recomputes the GSO after every step, which
# is fatal here because the basis entries are N**m-sized (~1500-bit) integers.
# --------------------------------------------------------------------------- #
def lll_reduce(basis, delta=Rational(3, 4)):
    """LLL-reduce a list of integer row vectors; returns the reduced integer
    basis as a list of int rows (shortest-ish vector first), or None if sympy's
    LLL cannot reduce this basis.

    NOTE: sympy's ddm_lll has an internal size-reduction ``assert`` that can
    fire (AssertionError) on some large/near-degenerate bases -- exactly the
    ones that arise near the N**(1/4) Coppersmith edge.  We retry a few delta
    values and, failing that, return None so the caller degrades gracefully
    instead of crashing."""
    if not basis:
        return basis
    M = Matrix([[int(v) for v in row] for row in basis])
    for d in (delta, Rational(99, 100)):
        try:
            reduced = M.lll(delta=d)
        except AssertionError:
            continue
        return [[int(reduced[i, j]) for j in range(reduced.cols)]
                for i in range(reduced.rows)]
    return None


# --------------------------------------------------------------------------- #
# Lattice-parameter selection.
# --------------------------------------------------------------------------- #
def _achievable_bits(N_bits, beta, m, t):
    """log2 of the largest recoverable X for the degree-1 Howgrave-Graham
    lattice {N^{m-i} f^i}_{i=0..m} + {x^j f^m}_{j=1..t}."""
    w = m + 1 + t
    Sx = m * (m + 1) // 2 + t * m + t * (t + 1) // 2  # sum of x-degrees on diag
    if Sx == 0:
        return 0.0
    # det = N^{m(m+1)/2} * X^{Sx};  HG condition ~ det^{1/w} < N^{beta*m}
    exp_of_N = (beta * m * w - m * (m + 1) / 2.0) / Sx
    return exp_of_N * N_bits


def choose_mt(N_bits, unknown_bits, beta=0.5, dim_cap=10, margin=1.5):
    """Smallest-dimension (m, t) whose HG bound covers `unknown_bits`, or None
    if infeasible within `dim_cap`.

    `dim_cap` is kept small (default 10) on purpose: sympy's LLL is fast and
    reliable on modest bases but slows sharply and can hit an internal
    assertion bug on the large lattices needed at the very N**(1/4) edge.  A
    dim<=10 lattice comfortably recovers a bit more than half of a 128-bit q's
    bits (sub-second) -- the practical Brumley-Boneh regime -- and larger
    demands simply return None so the caller degrades gracefully."""
    best = None
    for m in range(1, dim_cap):
        for t in range(0, 2 * m + 1):
            if m + 1 + t > dim_cap:
                break
            if _achievable_bits(N_bits, beta, m, t) >= unknown_bits + margin:
                dim = m + 1 + t
                if best is None or dim < best[0]:
                    best = (dim, m, t)
        if best is not None and best[0] <= m + 1:  # cannot get smaller
            break
    return (best[1], best[2]) if best else None


# --------------------------------------------------------------------------- #
# Coppersmith: recover a factor of N from its high bits.
# --------------------------------------------------------------------------- #
def factor_with_high_bits(N, a, X, beta=0.5, m=None, t=None, dim_cap=10):
    """Return a nontrivial factor q of N with a <= q < a + X (q = a + x0,
    0 <= x0 < X), or None.

    `a` is the known high part (the recovered prefix with the unknown low bits
    zeroed); `X` bounds the unknown tail (X = 2**unknown_bits).  Requires,
    classically, X < N**(beta**2)  (~ N**0.25 for balanced RSA)."""
    if a <= 1 or X <= 1:
        return a if (a > 1 and N % a == 0) else None
    # trivial: the tail is tiny -> the prefix itself already divides N
    if N % a == 0 and a > 1:
        return a

    N_bits = N.bit_length()
    unknown_bits = X.bit_length()  # ~ log2(X)
    if m is None or t is None:
        mt = choose_mt(N_bits, unknown_bits, beta=beta, dim_cap=dim_cap)
        if mt is None:
            return None                      # too many unknown bits for CM
        m, t = mt

    dim = m + 1 + t
    x = sympy.Symbol("x")

    # polynomials (as integer coefficient lists indexed by power of x)
    def fpow(i):  # (x + a)^i
        return [comb(i, k) * pow(a, i - k) for k in range(i + 1)]

    polys = []
    for i in range(m + 1):                    # N^{m-i} * f(x)^i
        c = fpow(i)
        row = [0] * dim
        Nfac = pow(N, m - i)
        for k, ck in enumerate(c):
            row[k] = ck * Nfac
        polys.append(row)
    fm = fpow(m)
    for j in range(1, t + 1):                 # x^j * f(x)^m
        row = [0] * dim
        for k, ck in enumerate(fm):
            row[k + j] = ck
        polys.append(row)

    # scale column `col` by X**col so lattice norm matches |h(xX)|
    Xpow = [pow(X, col) for col in range(dim)]
    B = [[row[col] * Xpow[col] for col in range(dim)] for row in polys]

    reduced = lll_reduce(B)
    if reduced is None:
        return None

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


if __name__ == "__main__":
    # Self-test: known q, hand Coppersmith only the top half of its bits.
    import time
    from bb_victim import generate_crt_key

    for bits, hide in ((256, 40), (256, 48), (256, 52), (256, 60)):
        key = generate_crt_key(bits)
        N, q = key["n"], key["q"]
        qb = q.bit_length()
        a = (q >> hide) << hide               # top (qb-hide) bits, low `hide` = 0
        X = 1 << hide
        mt = choose_mt(N.bit_length(), hide, beta=0.5)
        t0 = time.perf_counter()
        rec = factor_with_high_bits(N, a, X, beta=0.5)
        dt = time.perf_counter() - t0
        ok = (rec is not None and N % rec == 0)
        print("N=%d-bit q=%d-bit | hid low %2d bits (top %d known) | "
              "(m,t)=%s dim=%s | %6.2fs | %s"
              % (N.bit_length(), qb, hide, qb - hide, mt,
                 (mt[0] + 1 + mt[1]) if mt else None, dt,
                 "FACTORED (%s)" % ("q" if rec == q else "p")
                 if ok else "FAILED"))

