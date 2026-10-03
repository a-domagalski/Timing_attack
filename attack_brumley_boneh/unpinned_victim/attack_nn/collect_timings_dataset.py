"""
Timing-dataset collector for the ML/NN Brumley-Boneh attack.

Purpose
-------
Build a *training database* of the per-bit timing measurements that the
successful unpinned-victim wall-clock attack (``attack_remote.py`` full run)
consumes, so a model can be trained to replace / augment the hand-tuned
zero-one-gap decision.  It deliberately reuses the SAME measurement path as the
successful recovery:

  * the SAME victim script  -> ``bb_victim_server.py`` (separate, UNPINNED
    process, server-side ``--batch`` amplification);
  * the SAME attacker-side oracle -> ``attack_remote.RemoteVictim`` with the
    drift-cancelling INTERLEAVED-MIN estimator (``measure_pair``), driven through
    the unmodified ``attack_brumley_boneh.neighborhood_gap(..., interleave=True)``;
  * the SAME pinning discipline -> attacker pinned + warmed up, victim left on
    all cores (unpinned), exactly as in the successful run and the work log.

For each key we walk the *true* q prefix (known at profiling time, since we
generate the key) MSB->LSB, and at every recovered bit position we record the
measurement for the two candidate neighbourhoods (g = bit 0, ghi = bit 1),
labelled with the true bit.  This is the same walk ``diagnose_gap_separability``
does, so the recorded ``(Tg, Tghi, delta)`` are drawn from the identical
distribution the live attack sees.

What is recorded per (key, bit)
-------------------------------
  * ``true_bit``               -- the label (bit of q at this position).
  * ``meas[]``                 -- ``meas_per_bit`` independent interleaved-min
    measurements, each ``{Tg, Tghi, delta, neigh, queries, elapsed_s}`` (delta =
    Tg - Tghi is the exact statistic the attack thresholds).
  * ``exact`` (optional, on by default) -- the NOISE-FREE ground-truth leak for
    this bit: the Montgomery extra-reduction neighbourhood gap
    ``{ex_Tg, ex_Tghi, ex_delta}`` computed in-process on the same key via the
    instrumented ``CRTRSAVictim``.  This is a PROFILING-ONLY teacher signal (an
    auxiliary label / SNR reference); it is never available to the live remote
    attacker and must not be used as a model *input* at attack time.

Storage layout (append-friendly, one directory per parameter set)
-----------------------------------------------------------------
Every distinct measurement-parameter set gets its OWN directory, named by a
signature of the params that must match for samples to be poolable::

    nn_datasets/
      kb256_provable_rep64_ng128_s1_intlv_batch80_top40_mpb1/
        context.json      # the full parameter block + schema + host info
        dataset.jsonl     # ONE JSON object per key, appended (never rewritten)
        summary.json      # running counts (keys, bits, per-bit sample totals)
        0_<timestamp>/record.json   # per-key record, individually inspectable
        1_<timestamp>/record.json
        ...

``dataset.jsonl`` (JSON Lines) is the canonical store: enlarging the dataset on
a re-run is a pure append (no rewrite of prior data), so an interruption can at
worst leave a single trailing partial line, which the loader skips.  This is the
same "grow the dataset when run again" functionality as the sqam-only
``attack_nn`` collector, but corruption-proof by construction (that one needed a
salvage pass because it rewrote a single large ``dataset.json`` each time).

Reading it back (for the trainer)
---------------------------------
``load_dataset(param_dir)`` returns the per-key records; ``flatten_per_bit`` gives
a flat list of ``(features, label)`` samples for a per-bit distinguisher.  See
their docstrings.

Run (from the ``side-channel-attack-montMul`` root, using its venv)::

    venv\\Scripts\\python.exe attack_brumley_boneh\\unpinned_victim\\attack_nn\\collect_timings_dataset.py --keys 20
    venv\\Scripts\\python.exe attack_brumley_boneh\\unpinned_victim\\attack_nn\\collect_timings_dataset.py --config my_collect.json
    venv\\Scripts\\python.exe attack_brumley_boneh\\unpinned_victim\\attack_nn\\collect_timings_dataset.py smoke

COST WARNING: each bit costs ``neigh * repeat * 2`` round-trips (~3.4 ms each at
these params), so a top-40-bit key at repeat=64/neigh=128 is ~40 minutes.  Use
fewer top bits / smaller repeat/neigh for a quick starter set (or ``smoke``).
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))      # unpinned_victim/attack_nn/
_UNPINNED = os.path.dirname(_HERE)                       # unpinned_victim/
_ABB = os.path.dirname(_UNPINNED)                        # attack_brumley_boneh/
_ROOT = os.path.dirname(_ABB)                            # montMul/
for _p in (_HERE, _UNPINNED, _ABB, os.path.join(_ROOT, "attack_refined"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cpu_pin  # noqa: E402
import attack_brumley_boneh as abb  # noqa: E402
from bb_victim import generate_crt_key, save_crt_key, CRTRSAVictim  # noqa: E402
# Reuse the EXACT attacker-side client from the successful attack.
from attack_remote import RemoteVictim, connect, _free_port  # noqa: E402

# --------------------------------------------------------------------------- #
# Default configuration.  These mirror the SUCCESSFUL unpinned-victim run
# (work log 2026-09-21): interleaved-min, repeat=64, neigh=128, provable key,
# server_batch=80.  Only the dataset-specific knobs (num_keys, top_bits,
# meas_per_bit) are new.  Override any of them from a JSON config (see
# _CONFIG_KEYS) or the CLI.
# --------------------------------------------------------------------------- #
KEY_BITS = 256
KEY_SOURCE = "provable"       # "provable" (FIPS 186-5) | "random" (fast, tests)
SERVER_BATCH = 80             # victim-side decryptions/request (signal amplifier)

PIN_CORE = 0                  # attacker pins THIS core; victim stays unpinned
WARMUP_S = 1.5

REPEAT = 64                   # round-trips per measurement -> interleaved min
NEIGH = 128                   # neighbourhood-sum amplifier
SAMPLE = 1                    # keep 1 so the interleaved measure_pair path (the
                              # successful estimator) is used; >1 disables it.
INTERLEAVE = True             # drift-cancelling paired measurement (the fix)
ADAPTIVE_NEIGH = True         # ne = min(neigh, 2**i) near the top bits

TOP_BITS = 40                 # how many top bits of q to measure/label per key
MEAS_PER_BIT = 1              # independent interleaved-min measurements recorded
                              # per bit (>1 samples the noise distribution of the
                              # same label; multiplies cost linearly)
NUM_KEYS = 20                 # keys to add this run (dataset grows by this many)

COLLECT_EXACT = True          # also record the noise-free extra-reduction gap
                              # (profiling-only teacher signal / SNR reference)
                              # -> Pillar A (privileged-information distillation).
COLLECT_REFERENCE = True      # also record, per measurement, a FIXED
                              # key-independent reference round-trip (min over
                              # repeat) measured in the SAME time window as the
                              # pair.  It carries the victim's current turbo/DVFS
                              # speed with NO bit signal, so it is a clean drift
                              # covariate -> Pillar B (cross-time drift adaptation).
REF_CONST = 0xC0FFEE          # the fixed reference cipher base (mod n); any
                              # key-independent constant works -- only its
                              # VARIATION over the run (the drift) is used.
FORCE_VICTIM_UNPINNED = True  # after spawn, set the victim child to all cores so
                              # it stays unpinned even though the attacker is
                              # pinned (matches the successful run's end state)

OUT_BASE = os.path.join(_HERE, "nn_datasets")   # parent of the per-param dirs

_SCHEMA = "bb_timing_perbit_v2"

_CONFIG_KEYS = {
    "key_bits": "KEY_BITS", "key_source": "KEY_SOURCE",
    "server_batch": "SERVER_BATCH", "pin_core": "PIN_CORE", "warmup_s": "WARMUP_S",
    "repeat": "REPEAT", "neigh": "NEIGH", "sample": "SAMPLE",
    "interleave": "INTERLEAVE", "adaptive_neigh": "ADAPTIVE_NEIGH",
    "top_bits": "TOP_BITS", "meas_per_bit": "MEAS_PER_BIT", "num_keys": "NUM_KEYS",
    "collect_exact": "COLLECT_EXACT", "collect_reference": "COLLECT_REFERENCE",
    "ref_const": "REF_CONST",
    "force_victim_unpinned": "FORCE_VICTIM_UNPINNED", "out_base": "OUT_BASE",
}


# --------------------------------------------------------------------------- #
# Parameter signature -> directory name (samples are only poolable when these
# match, so each distinct set lives in its own directory).
# --------------------------------------------------------------------------- #
def param_signature():
    return "kb%d_%s_rep%d_ng%d_s%d_%s_batch%d_top%d_mpb%d" % (
        KEY_BITS, KEY_SOURCE, REPEAT, NEIGH, SAMPLE,
        "intlv" if INTERLEAVE else "block", SERVER_BATCH, TOP_BITS, MEAS_PER_BIT)


def _params_dict():
    return {
        "schema": _SCHEMA,
        "key_bits": KEY_BITS, "key_source": KEY_SOURCE,
        "server_batch": SERVER_BATCH, "pin_core": PIN_CORE, "warmup_s": WARMUP_S,
        "repeat": REPEAT, "neigh": NEIGH, "sample": SAMPLE,
        "interleave": INTERLEAVE, "adaptive_neigh": ADAPTIVE_NEIGH,
        "top_bits": TOP_BITS, "meas_per_bit": MEAS_PER_BIT,
        "collect_exact": COLLECT_EXACT, "collect_reference": COLLECT_REFERENCE,
        "ref_const": REF_CONST,
        "oracle": "external_roundtrip_min_interleaved",
        "measurement_fn": "attack_brumley_boneh.neighborhood_gap",
        "victim": "bb_victim_server.py (separate, unpinned)",
    }


# --------------------------------------------------------------------------- #
# Small robust JSON helpers (atomic context/summary; append-only jsonl).
# --------------------------------------------------------------------------- #
def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path, obj):
    """Append one record as a single line, flushed+fsynced so a crash can only
    truncate the trailing line (which the loader skips)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _count_existing_keys(jsonl_path):
    """Number of complete (parseable) key records already on disk."""
    if not os.path.exists(jsonl_path):
        return 0
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                n += 1
            except json.JSONDecodeError:
                break  # trailing partial line from an interrupted append
    return n


# --------------------------------------------------------------------------- #
# Victim process management (reuses the victim SCRIPT bb_victim_server.py).
# --------------------------------------------------------------------------- #
def spawn_victim(port, key_file):
    """Spawn the unpinned victim server on `port`, loading the pre-written key
    (``--reuse-key``).  Mirrors ``attack_remote.spawn_victim`` but takes the
    parameters explicitly (so a dataset run can vary them per directory)."""
    server = os.path.join(_UNPINNED, "bb_victim_server.py")
    proc = subprocess.Popen(
        [sys.executable, "-u", server, "--host", "127.0.0.1",
         "--port", str(port), "--key-file", key_file,
         "--key-bits", str(KEY_BITS), "--key-source", KEY_SOURCE,
         "--reuse-key", "--batch", str(SERVER_BATCH)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("victim exited early: %s"
                               % (proc.stdout.read() or "<no output>"))
        if line.startswith("READY"):
            return proc


def _set_all_cores(pid):
    """Force process `pid` onto ALL cores (undo any inherited pinning), so the
    victim stays UNPINNED even though the attacker (parent) is pinned.  Matches
    the successful run's end state.  Best-effort; returns True on success."""
    ncpu = os.cpu_count() or 1
    try:
        import psutil
        psutil.Process(pid).cpu_affinity(list(range(ncpu)))
        return True
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes
            PROCESS_SET_INFORMATION = 0x0200
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            h = k32.OpenProcess(PROCESS_SET_INFORMATION, False, int(pid))
            if h:
                k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE,
                                                       ctypes.c_size_t]
                k32.SetProcessAffinityMask(h, ctypes.c_size_t((1 << ncpu) - 1))
                k32.CloseHandle(h)
                return True
        except Exception:
            pass
    return False


def _kill(proc):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Per-key measurement: walk the TRUE q prefix, record each top bit.
# --------------------------------------------------------------------------- #
def measure_key(vic, key, out_dir, key_index, t0):
    """Measure the top TOP_BITS of q for one key over the interleaved-min oracle.

    Identical walk to ``diagnose_gap_separability``: at bit position ``i`` the
    correct bits ABOVE ``i`` form the prefix, ``g`` is the bit-0 candidate and
    ``ghi`` the bit-1 candidate; the label is the true bit ``(q >> i) & 1``.
    Returns the per-key record dict.
    """
    n = vic.n
    q = key["q"]
    qbits = (n.bit_length() + 1) // 2
    R = vic.q_radix()
    radix_inv = pow(R, -1, n)

    # Optional noise-free teacher: instrumented in-process victim on the SAME key.
    vic_exact = None
    radix_inv_ex = None
    if COLLECT_EXACT:
        vic_exact = CRTRSAVictim(key, timing="exact")
        R_ex = vic_exact.q_radix()
        radix_inv_ex = pow(R_ex, -1, n)

    # Fixed, key-independent reference cipher for the drift covariate (Pillar B).
    # Its round-trip time tracks the victim's current turbo/DVFS speed but has NO
    # dependence on the recovered bit, so its variation over the run is pure
    # drift.  Measured (min over repeat) in the SAME window as each pair.
    c_ref = (REF_CONST % n) if COLLECT_REFERENCE else None

    positions = [i for i in range(qbits - 2, qbits - 2 - TOP_BITS, -1) if i >= 0]

    bits = []
    for i in positions:
        prefix = (q >> (i + 1)) << (i + 1)     # correct bits above i
        g = prefix
        ghi = prefix | (1 << i)
        true_bit = (q >> i) & 1
        ne = min(NEIGH, max(1, 1 << i)) if ADAPTIVE_NEIGH else NEIGH

        meas = []
        for _ in range(MEAS_PER_BIT):
            tg, tghi = abb.neighborhood_gap(vic, g, ghi, radix_inv, n, ne,
                                            SAMPLE, interleave=INTERLEAVE)
            m = {
                "Tg": tg, "Tghi": tghi, "delta": tg - tghi,
                "neigh": ne, "queries": vic.queries,
                "elapsed_s": time.perf_counter() - t0,
            }
            if COLLECT_REFERENCE:
                # drift covariate: fixed-work reference round-trip, same window,
                # same min-over-repeat estimator the pair uses.
                m["ref"] = vic.measure(c_ref)
            meas.append(m)

        rec = {"bit_pos": i, "true_bit": true_bit, "neigh": ne, "meas": meas}

        if COLLECT_EXACT:
            ex_tg = abb.neighborhood_count(vic_exact, g, radix_inv_ex, n, ne)
            ex_tghi = abb.neighborhood_count(vic_exact, ghi, radix_inv_ex, n, ne)
            rec["exact"] = {"ex_Tg": ex_tg, "ex_Tghi": ex_tghi,
                            "ex_delta": ex_tg - ex_tghi}

        bits.append(rec)
        sys.stdout.write(
            "\r    key %d | bit %d/%d (pos %d, true=%d) | queries %d | %5.1fs   "
            % (key_index, len(bits), len(positions), i, true_bit,
               vic.queries, time.perf_counter() - t0))
        sys.stdout.flush()
    sys.stdout.write("\n")

    return {
        "key_index": key_index,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "n_hex": "%X" % n,
        "e": vic.e,
        "modulus_bits": n.bit_length(),
        "qbits": qbits,
        # q is the LABEL SOURCE (known because we generate the profiling key);
        # it is profiling-only ground truth, exactly like the attack's eval key.
        "q_hex": "%X" % q,
        "top_bits": len(positions),
        "key_queries": vic.queries,
        "key_seconds": time.perf_counter() - t0,
        "bits": bits,
    }


# --------------------------------------------------------------------------- #
# Main collection loop (append-only; grows the dataset on every run).
# --------------------------------------------------------------------------- #
def collect():
    out_dir = os.path.join(OUT_BASE, param_signature())
    os.makedirs(out_dir, exist_ok=True)
    ctx_path = os.path.join(out_dir, "context.json")
    jsonl_path = os.path.join(out_dir, "dataset.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    # context.json is authoritative for the params; guard against mixing.
    params = _params_dict()
    if os.path.exists(ctx_path):
        with open(ctx_path, encoding="utf-8") as f:
            old = json.load(f)
        mismatched = {k: (old.get(k), params[k]) for k in params
                      if k in old and old.get(k) != params[k]
                      and k not in ("victim", "oracle", "measurement_fn")}
        if mismatched:
            raise ValueError(
                "context.json in %s was written with different params: %s. "
                "Use a fresh directory (the signature should differ)."
                % (out_dir, mismatched))
    else:
        _atomic_write_json(ctx_path, {
            **params,
            "created": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
        })

    start_index = max(_count_existing_keys(jsonl_path), _next_subdir_index(out_dir))
    print("[collect] dir: %s" % out_dir)
    print("[collect] already have %d key(s); adding %d; params: %s"
          % (start_index, NUM_KEYS, param_signature()))

    # Pin the ATTACKER (this process) once, like the successful run.  Each victim
    # is spawned as a child and then forced back onto all cores so it stays
    # UNPINNED (the child would otherwise inherit our pinned affinity).
    cpu_pin.pin_process(core=PIN_CORE, high_priority=True, verbose=True)
    if WARMUP_S > 0:
        cpu_pin.warmup(WARMUP_S)

    added = 0
    for k in range(NUM_KEYS):
        key_index = start_index + k
        # 1) generate the profiling key (own modulus n; q is the secret we vary)
        key = generate_crt_key(KEY_BITS, source=KEY_SOURCE)
        key_dir = os.path.join(
            out_dir, "%d_%s" % (key_index,
                                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        os.makedirs(key_dir, exist_ok=True)
        key_file = os.path.join(key_dir, "victim_key.json")
        save_crt_key(key_file, key)

        # 2) spawn the UNPINNED victim on this key, force it onto all cores.
        port = _free_port()
        proc = spawn_victim(port, key_file)
        if FORCE_VICTIM_UNPINNED:
            ok = _set_all_cores(proc.pid)
            if not ok:
                print("  [warn] could not force victim to all cores "
                      "(victim may inherit attacker pinning)")

        # 3) connect the SAME RemoteVictim client used by the successful attack.
        t0 = time.perf_counter()
        try:
            sock = connect(port)
            vic = RemoteVictim(sock, key, repeat=REPEAT)
            print("\n[key %d/%d] n=%d bits | oracle=interleaved-min(repeat=%d) | "
                  "top_bits=%d" % (key_index, start_index + NUM_KEYS - 1,
                                   vic.n.bit_length(), REPEAT, TOP_BITS))
            record = measure_key(vic, key, out_dir, key_index, t0)
        finally:
            try:
                vic.close()
            except Exception:
                pass
            _kill(proc)

        # 4) persist: per-key record.json + append to the aggregate jsonl.
        _atomic_write_json(os.path.join(key_dir, "record.json"), record)
        _append_jsonl(jsonl_path, record)
        added += 1
        _write_summary(summary_path, jsonl_path)
        print("  saved key %d  (%d bits, %d queries, %.1fs)  -> %s"
              % (key_index, record["top_bits"], record["key_queries"],
                 record["key_seconds"], jsonl_path))

    print("\n[collect] done: added %d key(s); total now %d."
          % (added, _count_existing_keys(jsonl_path)))
    return out_dir


def _next_subdir_index(out_dir):
    mx = -1
    if os.path.isdir(out_dir):
        for d in os.listdir(out_dir):
            if os.path.isdir(os.path.join(out_dir, d)) and "_" in d:
                head = d.split("_", 1)[0]
                if head.isdigit():
                    mx = max(mx, int(head))
    return mx + 1


def _write_summary(summary_path, jsonl_path):
    n_keys = n_bits = n_meas = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break
            n_keys += 1
            for b in rec.get("bits", []):
                n_bits += 1
                n_meas += len(b.get("meas", []))
    _atomic_write_json(summary_path, {
        "keys": n_keys, "labelled_bits": n_bits, "bit_measurements": n_meas,
        "updated": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    })


# --------------------------------------------------------------------------- #
# Reading the dataset back (for the training script).
# --------------------------------------------------------------------------- #
def load_dataset(param_dir):
    """Yield the per-key records from ``param_dir/dataset.jsonl``.

    Skips a trailing partial line (interrupted append).  Also returns the params
    from ``context.json`` via the ``.params`` attribute of the returned list.
    """
    jsonl_path = os.path.join(param_dir, "dataset.jsonl")
    ctx_path = os.path.join(param_dir, "context.json")
    params = {}
    if os.path.exists(ctx_path):
        with open(ctx_path, encoding="utf-8") as f:
            params = json.load(f)
    records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    out = _RecordList(records)
    out.params = params
    return out


class _RecordList(list):
    """A plain list of key records with a ``.params`` attribute."""
    params = {}


def flatten_per_bit(records, include_exact=True):
    """Flatten key records into per-bit training samples.

    Returns a list of dicts, one per (key, bit, measurement)::

        {
          "features": {"delta": ..., "Tg": ..., "Tghi": ..., "neigh": ...,
                       "bit_pos": ...},
          "label": <true_bit 0/1>,
          "key_index": ..., "bit_pos": ...,
          "exact_delta": <noise-free extra-reduction gap or None>,  # aux target
        }

    ``features`` are attack-time observables only; ``exact_delta`` is a
    profiling-only auxiliary target (do NOT feed it as a model input at attack
    time).
    """
    samples = []
    for rec in records:
        for b in rec.get("bits", []):
            ex = b.get("exact", {}) if include_exact else {}
            for m in b.get("meas", []):
                samples.append({
                    "features": {
                        "delta": m.get("delta"),
                        "Tg": m.get("Tg"),
                        "Tghi": m.get("Tghi"),
                        "neigh": m.get("neigh"),
                        "bit_pos": b.get("bit_pos"),
                    },
                    "ref": m.get("ref"),               # drift covariate (Pillar B)
                    "elapsed_s": m.get("elapsed_s"),   # drift covariate (time)
                    "label": b.get("true_bit"),
                    "key_index": rec.get("key_index"),
                    "bit_pos": b.get("bit_pos"),
                    "exact_delta": ex.get("ex_delta"),
                })
    return samples


# --------------------------------------------------------------------------- #
# CLI / config.
# --------------------------------------------------------------------------- #
def _load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    applied = {}
    for k, v in cfg.items():
        if k.startswith("_") or v is None:
            continue
        g = _CONFIG_KEYS.get(k)
        if g is None:
            print("[config] WARNING: ignoring unknown key %r" % k)
            continue
        globals()[g] = v
        applied[k] = v
    print("[config] applied %d override(s) from %s" % (len(applied), path))
    return applied


def _apply_smoke():
    """Tiny, fast settings to verify the pipeline end-to-end."""
    global KEY_SOURCE, REPEAT, NEIGH, TOP_BITS, MEAS_PER_BIT, NUM_KEYS, WARMUP_S
    KEY_SOURCE = "random"      # fast keygen
    REPEAT = 4
    NEIGH = 4
    TOP_BITS = 3
    MEAS_PER_BIT = 1
    NUM_KEYS = 1
    WARMUP_S = 0.2


def main():
    ap = argparse.ArgumentParser(description="BB timing-dataset collector.")
    ap.add_argument("mode", nargs="?", default="collect",
                    choices=["collect", "smoke"])
    ap.add_argument("--config", default=None, help="JSON param overrides")
    ap.add_argument("--keys", type=int, default=None,
                    help="number of keys to add this run (overrides NUM_KEYS)")
    args = ap.parse_args()

    if args.config:
        _load_config(args.config)
    if args.mode == "smoke":
        _apply_smoke()
    if args.keys is not None:
        globals()["NUM_KEYS"] = args.keys

    if SAMPLE > 1:
        print("[warn] SAMPLE>1 disables the interleaved measure_pair estimator "
              "used by the successful run; set sample=1 to match it.")

    collect()


if __name__ == "__main__":
    main()
