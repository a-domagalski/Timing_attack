"""
Pinned real-oracle evaluation for the Brumley-Boneh attack.

Procedure (as prescribed by the work log's 2026-09-07 "Next steps"):
  1. Pin the CPU to a single core + high priority + warm up to a steady
     frequency (cpu_pin) -- the new lever meant to fight the multiplicative
     drift that floored the `real` oracle.
  2. Measure gap separability along the TRUE q prefix, BLOCK vs INTERLEAVE,
     under pinning -- the cheap instrument that tells us whether pinning lifted
     the AUC off the ~0.5 chance floor BEFORE spending a long recovery run.
  3. Run a full real-oracle recovery (beam + interleave + pinning) aimed at
     factoring N and recovering the whole key.

Run:  python run_pinned_eval.py   (via ..\\venv\\Scripts\\python.exe)
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABB = os.path.dirname(_HERE)                 # attack_brumley_boneh/ (shared modules + bb_cache/bb_results)
_ROOT = os.path.dirname(_ABB)                 # montMul/
for _p in (_HERE, _ABB, os.path.join(_ROOT, "attack_refined"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cpu_pin  # noqa: E402
import attack_brumley_boneh as abb  # noqa: E402
from bb_victim import (  # noqa: E402
    load_crt_key, generate_crt_key, save_crt_key, CRTRSAVictim,
)

# ------------------------------- config ------------------------------------ #
KEY_BITS = 256
PIN_CORE = 0
WARMUP_S = 1.5

# diagnostic (cheap AUC read under pinning)
DIAG_TOP_BITS = 24
DIAG_NEIGH = 100
DIAG_SAMPLE = 3
DIAG_TRIALS = 20
DIAG_BATCH = 120

# full recovery (only launched if the diagnostic looks worth it)
REC_NEIGH = 200
REC_SAMPLE = 5
REC_BATCH = 200
REC_BEAM = 16
REC_TAIL_BRUTE = 20
AUC_GO = 0.60          # interleave-AUC threshold to attempt full recovery
# --------------------------------------------------------------------------- #


def main():
    cpu_pin.pin_process(core=PIN_CORE, high_priority=True, verbose=True)
    if WARMUP_S > 0:
        print("[cpu_pin] warming up %.2fs to a steady frequency ..." % WARMUP_S)
        cpu_pin.warmup(WARMUP_S)

    cache = os.path.join(_ABB, "bb_cache", "keybits_%d_random" % KEY_BITS,
                         "key.json")
    if os.path.exists(cache):
        key = load_crt_key(cache)
        print("[key] loaded cached target: N=%d bits" % key["n"].bit_length())
    else:
        key = generate_crt_key(KEY_BITS)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        save_crt_key(cache, key)
        print("[key] generated new target: N=%d bits" % key["n"].bit_length())

    vic = CRTRSAVictim(key, timing="real", batch=DIAG_BATCH, repeat=1)
    R = vic.q_radix()
    rinv = pow(R, -1, vic.n)
    qbits = vic.q.bit_length()

    print("\n=== gap-separability diagnostic (PINNED, block vs interleave) ===")
    t0 = time.perf_counter()
    diag = abb.diagnose_gap_separability(
        vic, rinv, qbits, top_bits=DIAG_TOP_BITS, neigh=DIAG_NEIGH,
        sample=DIAG_SAMPLE, trials=DIAG_TRIALS, verbose=True)
    diag["diag_seconds"] = time.perf_counter() - t0
    diag["queries_after_diag"] = vic.queries
    print(json.dumps(diag, indent=2))

    block_auc = diag["block"]["auc"]
    inter_auc = diag["interleave"]["auc"]
    best_auc = max(block_auc, inter_auc)
    print("\n[diag] block AUC=%.3f  interleave AUC=%.3f  (chance=0.5)"
          % (block_auc, inter_auc))

    if best_auc < AUC_GO:
        print("[diag] AUC below %.2f -> the noise floor is NOT cleared even "
              "with pinning; full deep recovery would derail. Skipping the "
              "long recovery run (see work log)." % AUC_GO)
        print("[diag] Attempting a SHORT pinned real recovery (top bits only) "
              "to record Fig 3/4 separability under pinning instead.")
        abb.run(key_bits=KEY_BITS, timing="real", neigh=REC_NEIGH,
                sample=REC_SAMPLE, batch=REC_BATCH, repeat=1,
                recover_bits=24, tail_brute=REC_TAIL_BRUTE,
                beam_width=REC_BEAM, interleave=True, pin_cpu=True,
                pin_core=PIN_CORE, warmup_s=0.0, make_plots=True,
                use_key_cache=True, verbose=True)
        return

    print("[diag] AUC clears %.2f -> attempting FULL pinned real recovery "
          "(beam + interleave) to factor N." % AUC_GO)
    abb.run(key_bits=KEY_BITS, timing="real", neigh=REC_NEIGH,
            sample=REC_SAMPLE, batch=REC_BATCH, repeat=1,
            recover_bits=None, tail_brute=REC_TAIL_BRUTE,
            beam_width=REC_BEAM, interleave=True, pin_cpu=True,
            pin_core=PIN_CORE, warmup_s=0.0, make_plots=True,
            use_key_cache=True, verbose=True)


if __name__ == "__main__":
    main()
