"""
Montgomery-multiplication RSA timing attack (Schindler / Walter style).

It drives the victim process (main.py or the frozen main.exe) which decrypts
attacker-chosen ciphertexts with an *unprotected* square-and-multiply
Montgomery exponentiation and leaks a timing signal correlated with the number
of conditional ("extra") reductions performed by Montgomery multiplication.

Run
---
    python attack.py            # everything is configured in-code (see __main__)

There are NO command-line/environment knobs for the analysis: all parameters
(key size, oracle mode, sample count, beam width, caching, ...) are in-code
constants in the ``__main__`` block, mirroring
``side-channel-attack-sqam-only/attack/differential/attack_differential.py``.
The attacker launches the victim itself and passes the victim's own settings
(``--key-bits`` / ``--timing`` / ``--repeat`` / ``--reuse-key``) as CLI args.

What it produces (mirrors the differential attack)
--------------------------------------------------
Per timestamped results directory ``attack_results_<ts>/``:
  * ``run_<i>/figs/fig_for_bit_<j>.jpg``   per-bit distinguisher figure
  * ``run_<i>/figs/fig_data_bit_<j>.json`` the numbers behind each figure
  * ``run_<i>/results.json``               recovered key, per-bit results, diagnostics
  * ``run_<i>/measurement_times.json``     per-sample oracle timing (real mode)
  * ``summary.json``                       all runs combined
A per-(key-bits, mode) cache under ``mont_cache/`` persists the victim key and
the collected samples so re-runs skip the (slow, in real mode) collection and
extend it incrementally.

The private exponent d is recovered MSB-first.  For each candidate bit we
simulate the exact same Montgomery ladder the victim runs, on the exact same
base, and partition the sampled ciphertexts by whether the *following squaring*
would trigger an extra reduction under each hypothesis.  If the bit is 1 that
multiply actually happened, so the "extra reduction" group is measurably slower
(estimator A); comparing the two hypotheses cancels the base-dependent bias.

References (which part of the code implements which paper)
----------------------------------------------------------
* P. L. Montgomery, Math. Comp. 44 (1985).  -> the algorithm under attack; its
  conditional final subtraction ("extra reduction") in ``MontMul`` is the leak.
* P. Kocher, CRYPTO 1996.  -> recover the exponent bit-by-bit from timing.
* W. Schindler, CHES 2000, LNCS 1965.  -> the canonical Montgomery
  extra-reduction timing attack; the per-hypothesis decision here is his.
* C. D. Walter and S. Thompson, CT-RSA 2001.  -> extra reductions leak exponent
  digits (basis of estimator "A").
* J.-F. Dhem et al., CARDIS 1998, LNCS 1820.  -> the per-bit group-partitioning
  / average-time comparison realised here.
* D. Brumley and D. Boneh, USENIX Security 2003.  -> real-world demonstration of
  the same channel (motivates the "real" wall-clock oracle mode).
"""

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Montgomery arithmetic -- MUST match the victim (custom_rsa / main.py).
# Montgomery (1985): the `if r >= n: r -= n` final subtraction is the leak.
# --------------------------------------------------------------------------- #
BASE = 1 << 64


def limbsNr(x):
    return (x.bit_length() + 63) // 64


def getOmega(n):
    n0 = n & (BASE - 1)
    inv = pow(n0, -1, BASE)
    return (-inv) % BASE


def MontMul(x, y, n, omega):
    r = 0
    mask = BASE - 1
    mods = limbsNr(n)
    for _ in range(mods):
        yi = y & mask
        y >>= 64
        u = ((r + yi * x) & mask) * omega & mask
        r = (r + yi * x + u * n) >> 64
    if r >= n:
        return r - n, 1
    return r, 0


# --------------------------------------------------------------------------- #
# Terminal helpers (progress bar / ETA), matching the differential attack.
# --------------------------------------------------------------------------- #
def _fmt_dur(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m_, s = divmod(rem, 60)
    if h:
        return f"{h}h{m_:02d}m{s:02d}s"
    if m_:
        return f"{m_}m{s:02d}s"
    return f"{s}s"


def _bar_fill_chars():
    enc = (getattr(sys.stdout, "encoding", None) or "ascii").lower()
    try:
        "\u2588\u2591".encode(enc)
        return "\u2588", "\u2591"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"


def _progress_bar(done, total, width=28):
    frac = done / total if total else 1.0
    filled = int(round(frac * width))
    fill, empty = _bar_fill_chars()
    return (
        f"[{fill * filled}{empty * (width - filled)}] {done}/{total} {frac * 100:3.0f}%"
    )


# --------------------------------------------------------------------------- #
# Subprocess (victim) plumbing
# --------------------------------------------------------------------------- #
class Target:
    def __init__(self, argv):
        self.proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.n = int(self._readline(), 16)
        self.e = int(self._readline(), 16)
        self.queries = 0

    def _readline(self):
        line = self.proc.stdout.readline()
        if line == "":
            raise EOFError("target terminated unexpectedly")
        return line.strip()

    def decrypt(self, c):
        """Send ciphertext c, return (time, plaintext)."""
        self.queries += 1
        self.proc.stdin.write("{0:X}\n".format(c))
        self.proc.stdin.flush()
        t = float(self._readline())
        m = int(self._readline(), 16)
        return t, m

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()


# --------------------------------------------------------------------------- #
# Measurement collection (with progress bar + ETA + per-sample log)
# --------------------------------------------------------------------------- #
def collect_samples(target, sample_size, start_index=0, verbose=True):
    """Query the victim on random ciphertexts; return (cs, times, plaintexts,
    collect_log). `start_index` only shifts the printed sample counter so that an
    incremental extension of a cache reads naturally."""
    n = target.n
    cs, times, plaintexts, collect_log = [], [], [], []
    loop_start = time.perf_counter()
    for i in range(sample_size):
        c = random.randrange(1, n)
        t, m = target.decrypt(c)
        cs.append(c)
        times.append(t)
        plaintexts.append(m)
        done = i + 1
        elapsed = time.perf_counter() - loop_start
        collect_log.append(
            {"sample": start_index + done, "reported_time": float(t),
             "elapsed_s": float(elapsed)}
        )
        if verbose and (done % 100 == 0 or done == sample_size):
            avg = elapsed / done
            eta = avg * (sample_size - done)
            print(
                f"\r  {_progress_bar(done, sample_size)} | "
                f"elapsed {_fmt_dur(elapsed)} | ETA {_fmt_dur(eta)}   ",
                end="", flush=True,
            )
    if verbose:
        print(flush=True)
    return cs, times, plaintexts, collect_log


# --------------------------------------------------------------------------- #
# Persistence / caching.
# Re-running the attack (to retune the beam / regenerate plots) should NOT
# repeat the (slow, in real mode) collection.  When a `cache_dir` is given we
# persist, under it:
#   victim_key.txt   the key the victim generated/reloaded (n, e, d)
#   samples.json     the collected (c, time, plaintext) triples + a signature of
#                    every parameter they depend on; a mismatch re-collects.
# The sample list is grown incrementally: a re-run reuses the cached samples as a
# prefix and collects only the additional ones.  NOTE (real mode): a cached set
# freezes ONE noise realisation — ideal for iterating on the analysis, but for an
# independent noise-robustness study delete samples.json (or disable the cache).
# --------------------------------------------------------------------------- #
def _samples_signature(n, key_bits, timing, repeat):
    sig = {"n": str(n), "key_bits": int(key_bits), "timing": timing}
    if timing == "real":
        sig["repeat"] = int(repeat)
    return sig


def load_samples_cache(cache_dir, signature):
    path = os.path.join(cache_dir, "samples.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("signature") != signature:
        return None
    cs = [int(x, 16) for x in data["cs"]]
    times = [float(t) for t in data["times"]]
    plaintexts = [int(x, 16) for x in data["plaintexts"]]
    return cs, times, plaintexts


def save_samples_cache(cache_dir, signature, cs, times, plaintexts):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "samples.json")
    with open(path, "w") as f:
        json.dump(
            {
                "signature": signature,
                "cs": ["{0:X}".format(c) for c in cs],
                "times": [float(t) for t in times],
                "plaintexts": ["{0:X}".format(m) for m in plaintexts],
            },
            f,
        )


# --------------------------------------------------------------------------- #
# Distinguisher core (estimator A).
# --------------------------------------------------------------------------- #
def _bit_stats(res, base_bars, times, n, omega):
    """For the current per-ciphertext state `res`, look one step ahead: the
    always-present squaring, the conditional multiply, and the *following*
    squaring under each hypothesis.  Partition the measured `times` by whether
    that following squaring triggers an extra reduction and return the two
    hypothesis margins plus everything needed to plot / advance.

        delta1 = mean(time | following-sq reduces under bit=1) - mean(| not)
        delta0 = mean(time | following-sq reduces under bit=0) - mean(| not)
        margin = delta1 - delta0        (>0 => bit is 1)
    """
    ncts = len(res)
    enc0 = [0] * ncts
    enc1 = [0] * ncts
    e1_mask = np.zeros(ncts, dtype=bool)
    e0_mask = np.zeros(ncts, dtype=bool)
    for i in range(ncts):
        sq, _ = MontMul(res[i], res[i], n, omega)
        m1, _ = MontMul(sq, base_bars[i], n, omega)
        enc0[i], enc1[i] = sq, m1
        _, ex1 = MontMul(m1, m1, n, omega)   # following square if bit == 1
        _, ex0 = MontMul(sq, sq, n, omega)   # following square if bit == 0
        e1_mask[i] = bool(ex1)
        e0_mask[i] = bool(ex0)

    t = np.asarray(times, dtype=np.float64)

    def _delta(mask):
        hi, lo = t[mask], t[~mask]
        mh = hi.mean() if len(hi) else 0.0
        ml = lo.mean() if len(lo) else 0.0
        return float(mh - ml), float(mh), float(ml), int(len(hi)), int(len(lo))

    delta1, m1h, m1l, s1h, s1l = _delta(e1_mask)
    delta0, m0h, m0l, s0h, s0l = _delta(e0_mask)
    return {
        "delta1": delta1, "delta0": delta0, "margin": delta1 - delta0,
        "e1_mask": e1_mask, "e0_mask": e0_mask, "enc0": enc0, "enc1": enc1,
        "h1": (m1h, m1l, s1h, s1l), "h0": (m0h, m0l, s0h, s0l),
    }


def _auc(scores, labels):
    """ROC-AUC via Mann-Whitney U. 0.5 = no signal, 1.0 = perfect."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    u = ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- #
# Diagnostics: raw per-bit signal walked along the TRUE key (no error
# propagation) — isolates the measurement SNR, mirroring the differential
# attack's oracle_separability().
# --------------------------------------------------------------------------- #
def oracle_separability(true_d, cs, times, base_bars, n, omega, R_mod_n, unit):
    dbits = bin(true_d)[2:]
    res = [R_mod_n] * len(cs)
    # consume leading '1'
    for i in range(len(cs)):
        r, _ = MontMul(res[i], res[i], n, omega)
        r, _ = MontMul(r, base_bars[i], n, omega)
        res[i] = r

    margins, labels = [], []
    correct = 0
    for pos in range(1, len(dbits)):
        st = _bit_stats(res, base_bars, times, n, omega)
        margins.append(st["margin"])
        bit = int(dbits[pos])
        labels.append(bit)
        if (st["margin"] > 0) == bool(bit):
            correct += 1
        res = st["enc1"] if dbits[pos] == "1" else st["enc0"]

    margins, labels = np.array(margins), np.array(labels)
    auc = _auc(margins, labels)
    m1 = margins[labels == 1].mean() if (labels == 1).any() else float("nan")
    m0 = margins[labels == 0].mean() if (labels == 0).any() else float("nan")
    acc = correct / len(labels) if len(labels) else float("nan")
    print("\nOracle-prefix per-bit separability (estimator A on the TRUE key):")
    print(f"  mean margin  bit=1: {m1:+.4g}{unit}   bit=0: {m0:+.4g}{unit}")
    print(f"  per-bit sign accuracy: {correct}/{len(labels)} = {acc*100:.1f}%")
    print(f"  AUC(margin -> bit): {auc:.3f}   (0.5 = no signal, 1.0 = perfect)")
    return {
        "auc": auc,
        "mean_margin_bit1": float(m1),
        "mean_margin_bit0": float(m0),
        "true_prefix_per_bit_acc": float(acc),
    }


# --------------------------------------------------------------------------- #
# Recovery — exact reduction-count oracle (fixed-width beam over the exact
# two-sided invariant).  Unchanged algorithm; beam width is an in-code param.
# --------------------------------------------------------------------------- #
def recover_exact(cs, times, plaintexts, base_bars, n, omega, R_mod_n,
                  key_is_correct, verbose, beam_width=32):
    M = [int(t) for t in times]
    ncts = len(cs)
    max_bits = n.bit_length() + 8

    def avg(total, size):
        return total / size if size else 0.0

    def expand(res, cnt, cur_len):
        t1_hi = t1_lo = 0.0
        s1_hi = s1_lo = 0
        t0_hi = t0_lo = 0.0
        s0_hi = s0_lo = 0
        enc0 = [0] * ncts
        enc1 = [0] * ncts
        ex_sq = [0] * ncts
        ex_mul = [0] * ncts
        for i in range(ncts):
            sq, exs = MontMul(res[i], res[i], n, omega)
            m1, exm = MontMul(sq, base_bars[i], n, omega)
            enc0[i], enc1[i] = sq, m1
            ex_sq[i], ex_mul[i] = exs, exm
            _, e1 = MontMul(m1, m1, n, omega)
            _, e0 = MontMul(sq, sq, n, omega)
            if e1:
                t1_hi += M[i]; s1_hi += 1
            else:
                t1_lo += M[i]; s1_lo += 1
            if e0:
                t0_hi += M[i]; s0_hi += 1
            else:
                t0_lo += M[i]; s0_lo += 1
        margin = (avg(t1_hi, s1_hi) - avg(t1_lo, s1_lo)) \
            - (avg(t0_hi, s0_hi) - avg(t0_lo, s0_lo))

        remaining_cap = 2 * max(0, n.bit_length() - (cur_len + 1))
        children = []
        for bit in ("1", "0"):
            new_cnt = list(cnt)
            ok = True
            for i in range(ncts):
                new_cnt[i] += ex_sq[i]
                if bit == "1":
                    new_cnt[i] += ex_mul[i]
                if new_cnt[i] > M[i]:
                    ok = False
                    break
                if M[i] - new_cnt[i] > remaining_cap:
                    ok = False
                    break
            if ok:
                children.append((bit, enc1 if bit == "1" else enc0, new_cnt))
        return margin, children

    # Leading bit is 1; consume it (square then multiply).
    res0 = [0] * ncts
    cnt0 = [0] * ncts
    for i in range(ncts):
        r, ex = MontMul(R_mod_n, R_mod_n, n, omega)
        cnt0[i] += ex
        r, ex = MontMul(r, base_bars[i], n, omega)
        cnt0[i] += ex
        res0[i] = r

    beam = [[0.0, "1", res0, cnt0]]
    best_key = "1"
    depth = 1
    while beam:
        depth += 1
        if depth > max_bits:
            break
        candidates = []
        for score, key, res, cnt in beam:
            margin, children = expand(res, cnt, len(key))
            for bit, cres, ccnt in children:
                ckey = key + bit
                if key_is_correct(ckey):
                    if verbose:
                        print("  full exponent verified at {} bits".format(len(ckey)))
                    return ckey
                conf = margin if bit == "1" else -margin
                candidates.append([score + conf, ckey, cres, ccnt])
        if not candidates:
            break
        candidates.sort(key=lambda c: c[0], reverse=True)
        beam = candidates[:beam_width]
        if len(beam[0][1]) > len(best_key):
            best_key = beam[0][1]
        if verbose and depth % 8 == 0:
            print("  depth={} beam={} best_len={} top_score={:.3f}"
                  .format(depth, len(beam), len(best_key), beam[0][0]))

    return best_key


# --------------------------------------------------------------------------- #
# Recovery — noisy wall-clock oracle (greedy Schindler/Dhem decision).
# --------------------------------------------------------------------------- #
def recover_statistical(cs, times, plaintexts, base_bars, n, omega, R_mod_n,
                        key_is_correct, verbose):
    key = "1"
    res = [R_mod_n] * len(cs)
    for i in range(len(cs)):
        r, _ = MontMul(res[i], res[i], n, omega)
        r, _ = MontMul(r, base_bars[i], n, omega)
        res[i] = r
    max_bits = n.bit_length() + 8
    for _ in range(max_bits):
        st = _bit_stats(res, base_bars, times, n, omega)
        bit = "1" if st["margin"] > 0 else "0"
        key += bit
        res = st["enc1"] if bit == "1" else st["enc0"]
        if verbose and len(key) % 16 == 0:
            print("  recovered {} bits (delta1={:.4e} delta0={:.4e})"
                  .format(len(key), st["delta1"], st["delta0"]))
        if key_is_correct(key):
            if verbose:
                print("  full exponent verified at {} bits".format(len(key)))
            return key
    return key


# --------------------------------------------------------------------------- #
# Analyse the recovered key: walk it MSB-first, plotting the per-bit
# distinguisher and recording per-bit results vs the true key.
# --------------------------------------------------------------------------- #
def analyze_and_plot(recovered_key, true_d, cs, times, base_bars, n, omega,
                     R_mod_n, figures_dir, make_plots, unit, scale):
    dbits_true = bin(true_d)[2:]
    rbits = recovered_key
    res = [R_mod_n] * len(cs)
    for i in range(len(cs)):
        r, _ = MontMul(res[i], res[i], n, omega)
        r, _ = MontMul(r, base_bars[i], n, omega)
        res[i] = r

    bit_results = []
    correct = 1  # MSB (bit 0) is always 1
    for pos in range(1, len(rbits)):
        st = _bit_stats(res, base_bars, times, n, omega)
        decision = int(rbits[pos])
        actual = int(dbits_true[pos]) if pos < len(dbits_true) else None
        ok = (actual is not None and decision == actual)
        correct += int(ok)

        bit_results.append({
            "bit_position": pos,
            "actual_bit": actual,
            "predicted_bit": decision,
            "success": bool(ok),
            "delta1": st["delta1"],
            "delta0": st["delta0"],
            "margin": st["margin"],
        })

        if make_plots and figures_dir is not None:
            _plot_bit(pos, times, st, decision, actual, figures_dir, unit, scale)

        res = st["enc1"] if decision == 1 else st["enc0"]

    return bit_results, correct


def _plot_bit(pos, times, st, decision, actual, figures_dir, unit, scale):
    """Three-panel figure in the differential-attack style: the measured signal
    (top) and, for each hypothesis, the measured signal split by whether the
    following squaring triggers an extra reduction (bottom).  The correct
    hypothesis shows a visible gap between its two groups (delta); the wrong one
    does not."""
    t = np.asarray(times, dtype=np.float64) * scale
    e1 = st["e1_mask"]
    e0 = st["e0_mask"]
    n_bins = 30

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(t, bins=n_bins, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(t.mean(), color="navy", linestyle="--", linewidth=2,
                label=f"Mean: {t.mean():.3f}{unit}")
    ax1.set_xlabel(f"Measured decryption signal ({unit})")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Measured oracle signal (bit {pos})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    def _panel(ax, mask, delta, title):
        hi, lo = t[mask], t[~mask]
        if len(lo):
            ax.hist(lo, bins=n_bins, alpha=0.6, color="lightsteelblue",
                    edgecolor="black", label=f"no extra-red (n={len(lo)})")
            ax.axvline(lo.mean(), color="navy", linestyle="--", linewidth=2)
        if len(hi):
            ax.hist(hi, bins=n_bins, alpha=0.6, color="salmon",
                    edgecolor="black", label=f"extra-red (n={len(hi)})")
            ax.axvline(hi.mean(), color="darkred", linestyle="--", linewidth=2)
        ax.set_xlabel(f"Measured signal ({unit})")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title}   delta={delta * scale:+.3g}{unit}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    _panel(fig.add_subplot(gs[1, 0]), e0, st["delta0"],
           "H0 (bit = 0): group by following-square reduction")
    _panel(fig.add_subplot(gs[1, 1]), e1, st["delta1"],
           "H1 (bit = 1): group by following-square reduction")

    fig.suptitle(
        f"bit {pos}: margin={st['margin'] * scale:+.3g}{unit}  "
        f"pred={decision}  actual={actual}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"fig_for_bit_{pos}.jpg"))
    plt.close(fig)

    with open(os.path.join(figures_dir, f"fig_data_bit_{pos}.json"), "w") as f:
        json.dump({
            "bit_position": pos,
            "predicted_bit": decision,
            "actual_bit": actual,
            "delta1": st["delta1"],
            "delta0": st["delta0"],
            "margin": st["margin"],
            "h1_hi_mean_lo_mean_hi_n_lo_n": st["h1"],
            "h0_hi_mean_lo_mean_hi_n_lo_n": st["h0"],
            "unit": unit,
            "scale": scale,
        }, f)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def montgomery_attack(
    target_argv,
    key_bits,
    timing,
    repeat,
    num_samples,
    beam_width=32,
    cache_dir=None,
    use_sample_cache=False,
    figures_dir=None,
    make_plots=True,
    timing_out_path=None,
    verbose=True,
):
    """Full Montgomery extra-reduction attack against the subprocess victim.
    Returns (recovered_d, accuracy, bit_results, diag)."""
    print("Montgomery extra-reduction timing attack")
    print("=" * 60)
    print(f"Backend  : {timing}"
          + (f" (min of {repeat})" if timing == "real" else "")
          + f" | key_bits={key_bits} | samples={num_samples} | beam={beam_width}")

    # Victim key file (persisted so --reuse-key keeps the sample cache valid).
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key_file = os.path.join(cache_dir, "victim_key.txt")
    else:
        key_file = os.path.join(_HERE, "subprocess_key.txt")

    argv = list(target_argv) + [
        "--key-bits", str(key_bits),
        "--timing", timing,
        "--repeat", str(repeat),
        "--key-file", key_file,
    ]
    if use_sample_cache:
        argv.append("--reuse-key")

    print("Launching target:", " ".join(argv))
    target = Target(argv)
    n, e = target.n, target.e
    print(f"Public key: n={n.bit_length()} bits, e={e:X}")

    omega = getOmega(n)
    k = limbsNr(n)
    R = 1 << (64 * k)
    R_mod_n = R % n

    signature = _samples_signature(n, key_bits, timing, repeat)

    # Samples: reuse cache as a prefix and collect only what is missing.
    cached = None
    if cache_dir is not None and use_sample_cache:
        cached = load_samples_cache(cache_dir, signature)
    have = 0 if cached is None else len(cached[0])
    collect_log = None
    if cached is not None and have >= num_samples:
        cs = cached[0][:num_samples]
        times = cached[1][:num_samples]
        plaintexts = cached[2][:num_samples]
        note = " (frozen noise realisation)" if timing == "real" else ""
        print(f"  Reusing {num_samples} of {have} cached samples{note}; no queries.")
    else:
        if have > 0:
            print(f"  Extending cache: {have} reused, collecting {num_samples - have} "
                  f"new sample(s)...")
        else:
            print("Collecting samples from the oracle...")
        new_cs, new_t, new_p, collect_log = collect_samples(
            target, num_samples - have, start_index=have, verbose=verbose
        )
        if cached is not None:
            cs = cached[0] + new_cs
            times = cached[1] + new_t
            plaintexts = cached[2] + new_p
        else:
            cs, times, plaintexts = new_cs, new_t, new_p
        if cache_dir is not None and use_sample_cache:
            save_samples_cache(cache_dir, signature, cs, times, plaintexts)
            print(f"  Cached {len(cs)} samples -> "
                  f"{os.path.join(cache_dir, 'samples.json')}")

    target.close()
    base_bars = [(c * R) % n for c in cs]

    # True key (the victim saved it) — for diagnostics / accuracy / figures.
    true_d = None
    try:
        with open(key_file) as fh:
            true_d = int(fh.read().split("d:")[1].split("\n")[0].strip())
    except Exception as exc:
        print("  (could not read victim key for cross-check:", exc, ")")

    def key_is_correct(key):
        d = int(key, 2)
        for i in range(min(6, len(cs))):
            if pow(cs[i], d, n) != plaintexts[i]:
                return False
        return True

    exact_oracle = all(float(t).is_integer() and 0 <= t < 1e6 for t in times)
    unit = " reductions" if exact_oracle else " us"
    scale = 1.0 if exact_oracle else 1e6

    # Raw per-bit signal along the true key (SNR read, no error propagation).
    diag = {}
    if true_d is not None:
        diag = oracle_separability(true_d, cs, times, base_bars, n, omega,
                                   R_mod_n, unit)

    # Recovery.
    print("\nStarting recovery ...")
    if exact_oracle:
        print("  exact reduction-count oracle -> beam search")
        recovered_key = recover_exact(cs, times, plaintexts, base_bars, n, omega,
                                      R_mod_n, key_is_correct, verbose, beam_width)
    else:
        print("  real wall-clock oracle -> greedy statistical recovery")
        recovered_key = recover_statistical(cs, times, plaintexts, base_bars, n,
                                             omega, R_mod_n, key_is_correct, verbose)
    recovered_d = int(recovered_key, 2)

    # Per-bit figures + results (walk the recovered key).
    bit_results, correct = [], 1
    if true_d is not None:
        bit_results, correct = analyze_and_plot(
            recovered_key, true_d, cs, times, base_bars, n, omega, R_mod_n,
            figures_dir, make_plots, unit, scale
        )

    nbits = len(recovered_key)
    accuracy = correct / nbits if nbits else 0.0
    full = (true_d is not None and recovered_d == true_d)
    first_err = next((r["bit_position"] for r in bit_results if not r["success"]),
                     nbits)

    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  queries issued    : {target.queries}")
    print(f"  recovered d bits  : {nbits}")
    print(f"  recovered d (hex) : {recovered_d:X}")
    if true_d is not None:
        print(f"  true d (hex)      : {true_d:X}")
        print(f"  per-bit accuracy  : {correct}/{nbits} = {accuracy*100:.1f}%")
        print(f"  first error bit   : {first_err}")
        print(f"  full recovery     : {'YES (SUCCESS)' if full else 'no (MISMATCH)'}")
    diag["full_recovery"] = bool(full)
    diag["first_error_bit"] = int(first_err)
    diag["queries"] = int(target.queries)

    # Per-sample oracle timing -> separate json (real mode).
    if collect_log is not None and timing_out_path is not None:
        secs = [r["reported_time"] for r in collect_log]
        with open(timing_out_path, "w") as f:
            json.dump({
                "timing": timing,
                "key_bits": key_bits,
                "num_samples": len(collect_log),
                "repeat": repeat,
                "summary": {
                    "count": len(secs),
                    "mean": float(np.mean(secs)),
                    "std": float(np.std(secs)),
                    "min": float(np.min(secs)),
                    "max": float(np.max(secs)),
                },
                "per_sample": collect_log,
            }, f, indent=2)
        print(f"  saved per-sample timing -> {timing_out_path}")

    return recovered_d, accuracy, bit_results, diag


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # In-code configuration (no environment variables / CLI knobs).
    # ------------------------------------------------------------------ #
    # How to launch the victim.  Either drive main.py through this venv's
    # interpreter, or point at the frozen exe.  The analysis-side settings
    # (key size, oracle mode, ...) are configured here and passed to the victim.
    PYTHON = os.path.join(os.path.dirname(_HERE), "venv", "Scripts", "python.exe")
    MAIN_PY = os.path.join(os.path.dirname(_HERE), "main.py")
    TARGET_ARGV = [PYTHON, MAIN_PY]
    # TARGET_ARGV = [os.path.join(_HERE, "main.exe")]   # frozen victim instead

    KEY_BITS = 128            # multiple of 64 so n fills the top limb (signal exists)
    TIMING = "exact"          # "exact" (noise-free PoC) | "real" (wall-clock attack)
    MONT_REPEAT = 5           # real mode only: victim reports min over this many runs
    NUM_SAMPLES = 12000       # ciphertexts to collect / query
    BEAM_WIDTH = 32           # exact-mode beam width
    NUM_RUNS_ATTACK = 1       # independent full-attack repetitions
    MAKE_PLOTS = True
    USE_SAMPLE_CACHE = True   # persist+reuse victim key & samples across runs

    CACHE_ROOT = os.path.join(_HERE, "mont_cache")
    CACHE_DIR = os.path.join(CACHE_ROOT, f"keybits_{KEY_BITS}_{TIMING}")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(_HERE, "attack_mont_results", f"attack_results_{ts}")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for i in range(NUM_RUNS_ATTACK):
        print(f"\n{'=' * 50}\nRUN {i + 1}/{NUM_RUNS_ATTACK}\n{'=' * 50}")
        run_dir = os.path.join(results_dir, f"run_{i + 1}")
        figs_dir = os.path.join(run_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)

        recovered, accuracy, bit_results, diag = montgomery_attack(
            target_argv=TARGET_ARGV,
            key_bits=KEY_BITS,
            timing=TIMING,
            repeat=MONT_REPEAT,
            num_samples=NUM_SAMPLES,
            beam_width=BEAM_WIDTH,
            cache_dir=CACHE_DIR,
            use_sample_cache=USE_SAMPLE_CACHE,
            figures_dir=figs_dir,
            make_plots=MAKE_PLOTS,
            timing_out_path=os.path.join(run_dir, "measurement_times.json"),
        )

        run_result = {
            "recovered_key": recovered,
            "accuracy": float(accuracy),
            "timing": TIMING,
            "key_bits": KEY_BITS,
            "num_samples": NUM_SAMPLES,
            "beam_width": BEAM_WIDTH,
            "repeat": MONT_REPEAT,
            "diagnostics": diag,
            "bit_results": bit_results,
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(run_result, f, indent=2)
        all_results.append(run_result)

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {results_dir}")
