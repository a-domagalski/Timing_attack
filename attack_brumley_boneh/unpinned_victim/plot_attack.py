"""
Plot the results of a remote Brumley-Boneh timing-attack run (schema
`bb_remote_full_v2`, written by attack_remote.py `full` mode).

Produces clear, labelled, academic-style figures under <run_dir>/plots/:

    01_zero_one_gap.png     - the per-bit zero-one gap Delta = T(g) - T(g_hi)
                              vs the decision threshold tau, coloured by the TRUE
                              bit, with correct/incorrect decisions marked.
    02_recovery_accuracy.png- per-bit correctness + cumulative accuracy, with the
                              trustworthy top-prefix (leading-correct) highlighted.
    03_timing_cost.png      - cumulative wall-clock time + query count vs recovered
                              bit, and the time spent PER bit (how cost grows).
    04_winning_leaf.png     - ONLY when the top-scoring ("baseline") path failed
                              but a lower-ranked BEAM LEAF factored N: overlays the
                              baseline vs the winning leaf, plus the leading-correct
                              of every leaf (winner highlighted).

Usage (from the project root, using the venv python):
    python attack_brumley_boneh/unpinned_victim/plot_attack.py            # newest run
    python attack_brumley_boneh/unpinned_victim/plot_attack.py <run_dir>  # or results.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABB = os.path.dirname(_HERE)
_RUNS = os.path.join(_ABB, "bb_results", "remote_unpinned_victim")

# --- academic-ish style (colour-blind-friendly, gridded, high-dpi) --------- #
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linestyle": "--",
    "axes.axisbelow": True, "legend.frameon": True, "legend.framealpha": 0.92,
})
C_CORRECT = "#2ca02c"   # green
C_WRONG   = "#d62728"   # red
C_BIT0    = "#1f77b4"   # blue  (true bit = 0, large gap expected)
C_BIT1    = "#ff7f0e"   # orange(true bit = 1, ~flat gap expected)
C_TAU     = "#444444"   # grey  (threshold)
C_TIME    = "#1f77b4"   # blue
C_QUERY   = "#8c564b"   # brown
C_MAIN    = "#0b3d91"   # navy  (baseline / top path)
C_LEAF    = "#c1121f"   # crimson(winning leaf)
C_MUTE    = "#b0b0b0"   # grey  (other leaves)


# --------------------------------------------------------------------------- #
def _find_run(arg):
    """Resolve arg -> (run_dir, results.json path)."""
    if arg:
        p = os.path.abspath(arg)
        if os.path.isdir(p):
            return p, os.path.join(p, "results.json")
        if os.path.isfile(p):
            return os.path.dirname(p), p
        sys.exit("no such path: %s" % p)
    # newest remote_*_kb* dir with a results.json
    cands = []
    for d in os.listdir(_RUNS):
        rj = os.path.join(_RUNS, d, "results.json")
        if d.startswith("remote_") and os.path.isfile(rj):
            cands.append((os.path.getmtime(rj), os.path.join(_RUNS, d), rj))
    if not cands:
        sys.exit("no runs with results.json under %s" % _RUNS)
    cands.sort()
    return cands[-1][1], cands[-1][2]


def _col(per_bit, key):
    return np.array([r.get(key) if r.get(key) is not None else np.nan
                     for r in per_bit], dtype=float)


def _reconstruct_time(elapsed):
    """elapsed_s is per-session (resets on resume); rebuild a monotone
    cumulative wall-time across sessions and flag the resume boundaries."""
    cum, boundaries, offset, prev = [], [], 0.0, None
    for e in elapsed:
        if prev is not None and e < prev - 1e-9:     # session restart
            offset += prev
            boundaries.append(len(cum))              # index where resume happened
        cum.append(e + offset)
        prev = e
    return np.array(cum), boundaries


def _param_tag(P):
    return ("repeat=%s  neigh=%s  beam=%s  tail_brute=%s  interleave=%s  "
            "key=%s-%sb" % (P.get("repeat"), P.get("neigh"), P.get("beam_width"),
                            P.get("tail_brute"), P.get("interleave"),
                            P.get("key_source"), P.get("modulus_bits")))


def _outcome_tag(R):
    if R.get("factored"):
        return "OUTCOME: N FACTORED (d %s) via %s" % (
            "matches" if R.get("d_recovered_matches") else "MISMATCH",
            R.get("completion_method"))
    return "OUTCOME: not factored"


def _secondary_bit_axis(ax, per_bit):
    """Add a top x-axis giving the q-bit POSITION for each recovered index."""
    if not per_bit:
        return
    b0 = per_bit[0]["bit"]                 # MSB position at recovered index 1
    fwd = lambda x: b0 - (x - 1)           # index -> q-bit position
    inv = lambda b: b0 - b + 1
    sec = ax.secondary_xaxis("top", functions=(fwd, inv))
    sec.set_xlabel("q-bit position (MSB = %d)" % b0)


# --------------------------------------------------------------------------- #
def plot_gap(run, R, per_bit, outdir):
    idx = np.arange(1, len(per_bit) + 1)
    delta = _col(per_bit, "delta")
    tau = _col(per_bit, "tau")
    actual = _col(per_bit, "actual")
    correct = np.array([bool(r.get("correct")) for r in per_bit])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    # tau threshold as a step line
    ax.step(idx, tau, where="mid", color=C_TAU, lw=1.4, ls="--",
            label=r"decision threshold $\tau$")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)

    for bit_val, col in ((0, C_BIT0), (1, C_BIT1)):
        m = actual == bit_val
        ok = m & correct
        bad = m & ~correct
        ax.scatter(idx[ok], delta[ok], s=40, c=col, edgecolors="black",
                   linewidths=0.4, marker="o", zorder=3)
        ax.scatter(idx[bad], delta[bad], s=95, c=col, edgecolors=C_WRONG,
                   linewidths=1.8, marker="X", zorder=4)

    fe = R.get("first_error_bit")
    lead = R.get("leading_correct_bits")
    if lead:
        ax.axvspan(0.5, lead + 0.5, color=C_CORRECT, alpha=0.07,
                   label="leading-correct top prefix (%d bits)" % lead)
    if fe is not None and per_bit:
        b0 = per_bit[0]["bit"]
        ax.axvline(b0 - fe + 1, color=C_WRONG, ls=":", lw=1.6,
                   label="first error (q-bit %d)" % fe)

    ax.set_xlabel("recovered bit index  (1 = MSB of q, decreasing)")
    ax.set_ylabel(r"zero-one gap  $\Delta = T(g) - T(g_{\mathrm{hi}})$  [s]")
    ax.set_title("Per-bit zero-one gap and decisions\n%s" % _param_tag(R["params"]))
    _secondary_bit_axis(ax, per_bit)

    handles = [
        Line2D([], [], marker="o", ls="", mfc=C_BIT0, mec="black",
               label="true bit = 0 (correct)"),
        Line2D([], [], marker="o", ls="", mfc=C_BIT1, mec="black",
               label="true bit = 1 (correct)"),
        Line2D([], [], marker="X", ls="", mfc=C_MUTE, mec=C_WRONG, mew=1.6,
               ms=10, label="incorrect decision"),
        Line2D([], [], color=C_TAU, ls="--", label=r"threshold $\tau$"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2)
    fig.text(0.01, 0.005, "%s | %s" % (run, _outcome_tag(R)), fontsize=8,
             color="#333333")
    p = os.path.join(outdir, "01_zero_one_gap.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_accuracy(run, R, per_bit, outdir):
    idx = np.arange(1, len(per_bit) + 1)
    correct = np.array([bool(r.get("correct")) for r in per_bit])
    cum_acc = np.cumsum(correct) / idx * 100.0

    fig, (axs, ax) = plt.subplots(
        2, 1, figsize=(11, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.08})

    # correctness strip
    axs.scatter(idx[correct], np.zeros(correct.sum()), marker="s", s=26,
                c=C_CORRECT, label="correct")
    axs.scatter(idx[~correct], np.zeros((~correct).sum()), marker="s", s=26,
                c=C_WRONG, label="wrong")
    axs.set_yticks([]); axs.set_ylabel("per-bit", rotation=0, ha="right", va="center")
    axs.legend(loc="center left", ncol=2, bbox_to_anchor=(0.0, 0.5))
    axs.grid(False)

    ax.plot(idx, cum_acc, color=C_MAIN, lw=2.0, label="cumulative accuracy")
    ax.axhline(50, color=C_WRONG, ls=":", lw=1.2, label="chance (50%)")
    lead = R.get("leading_correct_bits")
    if lead:
        ax.axvspan(0.5, lead + 0.5, color=C_CORRECT, alpha=0.10,
                   label="leading-correct (%d bits)" % lead)
    ax.set_ylim(0, 105)
    ax.set_xlabel("recovered bit index  (1 = MSB of q)")
    ax.set_ylabel("cumulative top-path accuracy  [%]")
    ax.legend(loc="lower left", ncol=2)

    txt = ("bits: %s/%s correct (%.1f%%)\nleading-correct: %s top bits\n"
           "first error: q-bit %s\nearly-stopped: %s\n%s"
           % (R.get("bits_correct"), R.get("bits_total"),
              100.0 * R.get("bits_correct", 0) / max(1, R.get("bits_total", 1)),
              lead, R.get("first_error_bit"), R.get("early_stopped"),
              _outcome_tag(R)))
    ax.text(0.985, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#999999",
                                  alpha=0.9))
    axs.set_title("Recovery correctness per bit and cumulative accuracy\n%s"
                  % _param_tag(R["params"]))
    p = os.path.join(outdir, "02_recovery_accuracy.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_timing(run, R, per_bit, outdir):
    idx = np.arange(1, len(per_bit) + 1)
    elapsed = _col(per_bit, "elapsed_s")
    queries = _col(per_bit, "queries")
    cum_t, boundaries = _reconstruct_time(elapsed)
    cum_h = cum_t / 3600.0
    per_bit_t = np.diff(cum_t, prepend=0.0)
    per_bit_t[per_bit_t < 0] = np.nan

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(11, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

    # Panel A: cumulative time + cumulative queries (twin axis)
    axA.plot(idx, cum_h, color=C_TIME, lw=2.0, marker=".", ms=4,
             label="cumulative time")
    axA.set_ylabel("cumulative wall-clock time  [h]", color=C_TIME)
    axA.tick_params(axis="y", labelcolor=C_TIME)
    axq = axA.twinx()
    axq.plot(idx, queries / 1e6, color=C_QUERY, lw=1.6, ls="--",
             label="cumulative queries")
    axq.set_ylabel("cumulative round-trips  [millions]", color=C_QUERY)
    axq.tick_params(axis="y", labelcolor=C_QUERY)
    axq.grid(False)
    for j, b in enumerate(boundaries):
        axA.axvline(b + 0.5, color="#9467bd", ls=":", lw=1.4,
                    label="resume boundary" if j == 0 else None)
    axA.set_title("Attack cost vs recovered bit  (total %.2f h, %.2f M round-trips)"
                  "\n%s" % (cum_h[-1] if len(cum_h) else 0,
                           (queries[-1] / 1e6) if len(queries) else 0,
                           _param_tag(R["params"])))
    lA, laA = axA.get_legend_handles_labels()
    lq, lqA = axq.get_legend_handles_labels()
    axA.legend(lA + lq, laA + lqA, loc="upper left")

    # Panel B: time spent PER bit
    axB.bar(idx, per_bit_t, width=0.85, color=C_TIME, alpha=0.8,
            edgecolor="black", linewidth=0.3)
    for j, b in enumerate(boundaries):
        axB.axvline(b + 0.5, color="#9467bd", ls=":", lw=1.4)
    axB.set_xlabel("recovered bit index  (1 = MSB of q)")
    axB.set_ylabel("time for this bit  [s]")
    axB.set_title("Per-bit recovery time (rises then plateaus as the beam fills)")
    _secondary_bit_axis(axA, per_bit)
    p = os.path.join(outdir, "03_timing_cost.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_winning_leaf(run, R, per_bit, outdir):
    """Only meaningful when the top path failed but a lower-ranked leaf factored."""
    leaves = R.get("beam_leaves") or []
    full = R.get("beam_leaves_full")
    comp = R.get("_completion") or {}
    win = comp.get("winning_candidate")
    if win is None:
        # fall back: the highest leading_correct leaf
        if leaves:
            win = int(np.argmax([lf.get("leading_correct", 0) for lf in leaves]))
        else:
            return None
    baseline_lead = R.get("leading_correct_bits", 0)
    win_lead = leaves[win].get("leading_correct") if win < len(leaves) else None
    non_top = win != 0

    ncols = 2 if (full and non_top and win < len(full)) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11 if ncols == 2 else 6.8, 4.8))
    axes = np.atleast_1d(axes)

    # left (optional): baseline vs winning-leaf gap traces
    if ncols == 2:
        axL = axes[0]
        mpb = per_bit
        lpb = full[win]["per_bit"]
        xi = np.arange(1, len(mpb) + 1)
        xl = np.arange(1, len(lpb) + 1)
        axL.plot(xi, _col(mpb, "delta"), color=C_MAIN, lw=1.6, marker=".",
                 ms=4, label="baseline (top path)")
        axL.plot(xl, _col(lpb, "delta"), color=C_LEAF, lw=1.6, marker=".",
                 ms=4, alpha=0.85, label="winning leaf (rank %d)" % win)
        axL.step(xi, _col(mpb, "tau"), where="mid", color=C_TAU, ls="--",
                 lw=1.0, label=r"$\tau$ (baseline)")
        axL.axhline(0, color="black", lw=0.8, alpha=0.5)
        if R.get("first_error_bit") is not None:
            b0 = mpb[0]["bit"]
            axL.axvline(b0 - R["first_error_bit"] + 1, color=C_MAIN, ls=":",
                        lw=1.4, label="baseline first error")
        axL.set_xlabel("recovered bit index")
        axL.set_ylabel(r"zero-one gap $\Delta$  [s]")
        axL.set_title("Baseline vs winning leaf: gap traces")
        axL.legend(loc="upper right", fontsize=9)
        axBar = axes[1]
    else:
        axBar = axes[0]

    # leading-correct per leaf (winner highlighted)
    lead_vals = [lf.get("leading_correct", 0) for lf in leaves]
    ranks = np.arange(len(lead_vals))
    cols = [C_LEAF if i == win else C_MUTE for i in ranks]
    axBar.bar(ranks, lead_vals, color=cols, edgecolor="black", linewidth=0.3)
    axBar.axhline(baseline_lead, color=C_MAIN, ls="--", lw=1.4,
                  label="baseline leading-correct (%d)" % baseline_lead)
    u_used = comp.get("unknown_bits_used")
    if u_used is not None and per_bit:
        need = (per_bit[0]["bit"] + 1) - u_used   # correct top bits Coppersmith used
        axBar.axhline(need - 1, color=C_CORRECT, ls=":", lw=1.4,
                      label="Coppersmith threshold (~%d)" % (need - 1))
    axBar.set_xlabel("beam leaf (0 = best score)")
    axBar.set_ylabel("leading-correct recovered bits")
    ttl = "Leading-correct per beam leaf"
    if comp.get("factored"):
        ttl += "  -  leaf %d FACTORED (u=%s)" % (win, u_used)
    axBar.set_title(ttl)
    axBar.legend(loc="upper right", fontsize=9)

    note = ""
    if non_top and comp.get("factored"):
        note = ("The top-scoring path failed (%d correct top bits) but leaf #%d "
                "carried %s correct top bits and factored N." %
                (baseline_lead, win, win_lead))
    elif full is None:
        note = ("per-leaf per-bit series not saved for this run "
                "(pre-SAVE_ALL_LEAF_RECORDS); showing leaf summary only.")
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="#333333")
    fig.suptitle("Winning beam leaf analysis  -  %s" % _outcome_tag(R), y=1.02,
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, "04_winning_leaf.png")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    arg = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    run_dir, rj = _find_run(arg)
    with open(rj) as fh:
        R = json.load(fh)
    # attach completion_output.json (which leaf won) if present
    co = os.path.join(run_dir, "completion_output.json")
    if os.path.isfile(co):
        try:
            R["_completion"] = json.load(open(co))
        except Exception:
            R["_completion"] = None
    per_bit = R.get("per_bit") or []
    if not per_bit:
        sys.exit("results.json has no per_bit series: %s" % rj)

    run = os.path.basename(run_dir)
    outdir = os.path.join(run_dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    made = [plot_gap(run, R, per_bit, outdir),
            plot_accuracy(run, R, per_bit, outdir),
            plot_timing(run, R, per_bit, outdir)]
    leaf = plot_winning_leaf(run, R, per_bit, outdir)
    if leaf:
        made.append(leaf)

    print("run: %s" % run_dir)
    print("outcome: %s" % _outcome_tag(R))
    print("wrote %d figures to %s:" % (len(made), outdir))
    for p in made:
        print("  - %s" % os.path.basename(p))


if __name__ == "__main__":
    main()
