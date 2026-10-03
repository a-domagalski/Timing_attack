"""
Noise-attribution probe for the Brumley-Boneh `real` timing oracle.

Question it answers (empirically, not by assumption): of all the things that can
perturb a wall-clock measurement on a modern OoO CPU -- CPU/frequency &
power management, cache/branch/speculation, memory contention, SMT sibling
activity, interrupts / context switches / scheduling, core migration, timer
overhead -- WHICH family dominates the noise that floors our `real` recovery?
That decides whether the fixed-clock power-plan/BIOS lever is well-targeted.

Method
------
Every iteration we co-measure, back-to-back (via the C __rdtsc core), TWO
cycle counts over the SAME clock:

  * TARGET  T : one full CRT decryption of a FIXED ciphertext (the attack's own
                workload; p-half + q-half, batch-timed).  Its retired-instruction
                count is constant (fixed input), so all variation in T is noise.
  * REFERENCE R : a FIXED, unrelated modular exponentiation (also constant work).

Because BOTH have constant instruction counts, any variation is noise.  The key
discriminator:

  * A GLOBAL MULTIPLICATIVE factor (core frequency / DVFS / power state, thermal)
    scales EVERY workload's wall-time together -> T and R are highly correlated,
    and dividing them (T/R) CANCELS it, collapsing T's relative variance.
  * WORKLOAD-SPECIFIC noise (memory/cache/SMT sibling/contention specific to the
    decryption) does NOT appear in the fixed reference -> low T-R correlation,
    and T/R does NOT collapse the variance.
  * ONE-SIDED spikes (interrupts / context switches / scheduling) show up as a
    heavy upper tail: mean >> min, large fraction above median+k*MAD; a
    min-over-repeat estimator removes them (a different lever than fixing the clock).

We run for a multi-minute wall-time budget (not a fixed sample count) so slow
thermal/DVFS drift -- the thing that plagues the 80-min attack -- actually has
time to appear.  Pinning is applied first, so we characterise the SAME regime
the attack runs in.

Run:  python attribution_probe.py   (via ..\\venv\\Scripts\\python.exe)
Artifacts: bb_results/attribution_<ts>/report.json + figs.
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABB = os.path.dirname(_HERE)                 # attack_brumley_boneh/ (shared modules + bb_cache/bb_results)
_ROOT = os.path.dirname(_ABB)                 # montMul/
for _p in (_HERE, _ABB, os.path.join(_ROOT, "attack_refined"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cpu_pin  # noqa: E402
import montmul_c as mc  # noqa: E402
from bb_victim import load_crt_key, generate_crt_key, CRTRSAVictim  # noqa: E402

# ------------------------------- config ------------------------------------ #
KEY_BITS = 256
TIME_BUDGET_S = 180        # multi-minute window so slow drift can appear
BATCH = 100                # ladders timed together per measurement (attack-like)
PIN_CORE = 0
WARMUP_S = 1.5
# --------------------------------------------------------------------------- #


def _relstd(a):
    a = np.asarray(a, float)
    m = a.mean()
    return float(a.std(ddof=1) / m) if m else float("nan")


def _autocorr1(a):
    a = np.asarray(a, float)
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


def main():
    cpu_pin.pin_process(core=PIN_CORE, high_priority=True, verbose=True)
    if WARMUP_S > 0:
        print("[cpu_pin] warming up %.2fs ..." % WARMUP_S)
        cpu_pin.warmup(WARMUP_S)

    cache = os.path.join(_ABB, "bb_cache", "keybits_%d_random" % KEY_BITS,
                         "key.json")
    key = load_crt_key(cache) if os.path.exists(cache) else generate_crt_key(KEY_BITS)
    victim = CRTRSAVictim(key, timing="real", batch=BATCH, repeat=1)
    hz = mc.tsc_hz(0.2)

    n, p = key["n"], key["p"]
    c0 = 0x0123456789ABCDEF0123456789ABCDEF % n     # fixed target ciphertext
    ref_base, ref_exp, ref_mod = 3, key["d1"], p    # fixed reference workload

    print("[probe] pinned; hz=%.3e; batch=%d; budget=%ds; measuring T (fixed "
          "decrypt) vs R (fixed modexp) ..." % (hz, BATCH, TIME_BUDGET_S))

    T, R, ts = [], [], []
    t_start = time.perf_counter()
    last = t_start
    while True:
        now = time.perf_counter()
        if now - t_start >= TIME_BUDGET_S:
            break
        _, _, cyc_ref = mc.modexp(ref_base, ref_exp, ref_mod,
                                  do_time=True, batch=BATCH)
        info = victim.decrypt_instrumented(c0, do_time=True)
        cyc_tar = info["cyc_p"] + info["cyc_q"]
        R.append(float(cyc_ref))
        T.append(float(cyc_tar))
        ts.append(now - t_start)
        if now - last > 15:
            last = now
            sys.stdout.write("\r  elapsed %5.1fs | samples %d   "
                             % (now - t_start, len(T)))
            sys.stdout.flush()
    sys.stdout.write("\n")

    T = np.asarray(T)
    R = np.asarray(R)
    ts = np.asarray(ts)
    N = len(T)

    # --- core discriminators ------------------------------------------------ #
    corr = float(np.corrcoef(R, T)[0, 1]) if N > 2 else float("nan")
    ratio = T / R
    rel_T, rel_R, rel_ratio = _relstd(T), _relstd(R), _relstd(ratio)
    # fraction of T's relative variance explained by the common (freq-like) factor
    frac_common = float(1 - (rel_ratio ** 2) / (rel_T ** 2)) if rel_T else float("nan")

    # --- slow drift over the run (thirds of wall time) ---------------------- #
    thirds = np.array_split(np.arange(N), 3)
    block_means_T = [float(T[idx].mean()) for idx in thirds]
    drift_T = (max(block_means_T) - min(block_means_T)) / T.mean()
    block_means_ratio = [float(ratio[idx].mean()) for idx in thirds]
    drift_ratio = (max(block_means_ratio) - min(block_means_ratio)) / ratio.mean()

    # --- autocorrelation (slow correlated drift vs iid jitter) -------------- #
    ac_T, ac_ratio = _autocorr1(T), _autocorr1(ratio)

    # --- batch-averaging test: does more data cut the noise (~1/sqrt(k)) or
    #     plateau (correlated drift)? ; and min-of-k (one-sided rejection) ---- #
    avg_test = {}
    for k in (1, 5, 20, 50):
        m = (N // k) * k
        if m < k:
            continue
        groups = T[:m].reshape(-1, k)
        avg_test[str(k)] = {
            "relstd_mean_of_k": _relstd(groups.mean(axis=1)),
            "relstd_min_of_k": _relstd(groups.min(axis=1)),
        }

    # --- one-sided spike diagnostics (interrupts / scheduling) -------------- #
    med = float(np.median(T))
    mad = float(np.median(np.abs(T - med))) or 1.0
    frac_hi = float(np.mean(T > med + 3 * mad))
    mean_over_min = float(T.mean() / T.min())

    # --- absolute per-decryption timing (context) --------------------------- #
    us_per_decrypt = (T / BATCH) / hz * 1e6            # p+q ladders per "decrypt"
    mean_us, std_us = float(us_per_decrypt.mean()), float(us_per_decrypt.std())

    # --- verdict ------------------------------------------------------------ #
    freq_like = (corr > 0.5) and (frac_common > 0.5)
    spike_heavy = frac_hi > 0.02 or mean_over_min > 1.15
    if freq_like:
        verdict = ("DOMINANT NOISE = GLOBAL MULTIPLICATIVE factor shared across "
                   "workloads (frequency / DVFS / power / thermal). Corr(T,R)=%.2f "
                   "and dividing out the reference removes %.0f%% of T's relative "
                   "variance. => the fixed-clock power-plan/BIOS lever is well "
                   "targeted." % (corr, 100 * frac_common))
    else:
        verdict = ("DOMINANT NOISE is NOT a shared global factor (corr(T,R)=%.2f, "
                   "reference removes only %.0f%% of T's variance). It is largely "
                   "workload-specific (memory/cache/SMT/contention). => fixing the "
                   "clock alone will NOT be sufficient." % (corr, 100 * frac_common))
    if spike_heavy:
        verdict += (" ALSO: heavy one-sided tail (%.1f%% of samples > med+3MAD, "
                    "mean/min=%.3f) => interrupts/scheduling present; a "
                    "min-over-repeat estimator would help (orthogonal to the clock)."
                    % (100 * frac_hi, mean_over_min))

    report = {
        "config": {"key_bits": KEY_BITS, "time_budget_s": TIME_BUDGET_S,
                   "batch": BATCH, "pinned_core": PIN_CORE, "warmup_s": WARMUP_S,
                   "tsc_hz": hz, "samples": N},
        "target_rel_std": rel_T, "reference_rel_std": rel_R,
        "ratio_rel_std": rel_ratio, "corr_target_reference": corr,
        "frac_variance_common_multiplicative": frac_common,
        "slow_drift_target_fraction": drift_T,
        "slow_drift_ratio_fraction": drift_ratio,
        "autocorr_lag1_target": ac_T, "autocorr_lag1_ratio": ac_ratio,
        "batch_averaging_test": avg_test,
        "one_sided_frac_above_med_3mad": frac_hi,
        "mean_over_min": mean_over_min,
        "per_decrypt_us_mean": mean_us, "per_decrypt_us_std": std_us,
        "verdict": verdict,
    }

    ts_dir = os.path.join(_ABB, "bb_results", "inprocess_victim", "attribution",
                          "attribution_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(ts_dir, exist_ok=True)
    with open(os.path.join(ts_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    # figures
    try:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(ts, T / T.mean(), lw=0.6, color="#d62728", label="target (fixed decrypt)")
        ax.plot(ts, R / R.mean(), lw=0.6, color="#1f77b4", alpha=0.8, label="reference (fixed modexp)")
        ax.plot(ts, ratio / ratio.mean(), lw=0.6, color="green", alpha=0.8, label="target / reference")
        ax.set_xlabel("wall time (s)"); ax.set_ylabel("normalised time")
        ax.set_title("Noise attribution: if T and R move together, the noise is a "
                     "global (frequency) factor that cancels in T/R")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.savefig(os.path.join(ts_dir, "fig_timeseries.jpg"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(R / R.mean(), T / T.mean(), s=4, alpha=0.3)
        ax.set_xlabel("reference (norm)"); ax.set_ylabel("target (norm)")
        ax.set_title("target vs reference (corr=%.2f)" % corr); ax.grid(alpha=0.3)
        fig.savefig(os.path.join(ts_dir, "fig_scatter.jpg"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print("[probe] plotting skipped: %r" % exc)

    # terminal report
    print("=" * 78)
    print("NOISE ATTRIBUTION REPORT  (samples=%d, %.0fs, batch=%d, pinned)" % (N, TIME_BUDGET_S, BATCH))
    print("-" * 78)
    print("  per-decryption time      : %.2f us  (std %.2f us)" % (mean_us, std_us))
    print("  rel std  target T        : %.4f  (%.1f%%)" % (rel_T, 100 * rel_T))
    print("  rel std  reference R     : %.4f  (%.1f%%)" % (rel_R, 100 * rel_R))
    print("  rel std  ratio T/R       : %.4f  (%.1f%%)  <- common-mode removed" % (rel_ratio, 100 * rel_ratio))
    print("  corr(T, R)               : %+.3f" % corr)
    print("  variance explained by a  : %.1f%%" % (100 * frac_common))
    print("    shared multiplicative")
    print("    (frequency) factor")
    print("  slow drift over run  T   : %.1f%%   T/R : %.1f%%" % (100 * drift_T, 100 * drift_ratio))
    print("  autocorr lag-1  T        : %+.3f   T/R : %+.3f" % (ac_T, ac_ratio))
    print("  one-sided tail (>med+3MAD): %.2f%%   mean/min : %.3f" % (100 * frac_hi, mean_over_min))
    print("  batch-averaging (rel std of mean-of-k | min-of-k):")
    for k, d in avg_test.items():
        print("     k=%-3s  mean %.4f   min %.4f" % (k, d["relstd_mean_of_k"], d["relstd_min_of_k"]))
    print("-" * 78)
    print("  VERDICT: " + verdict)
    print("  artifacts: %s" % ts_dir)
    print("=" * 78)


if __name__ == "__main__":
    main()
