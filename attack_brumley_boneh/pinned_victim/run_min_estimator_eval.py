"""
Min-estimator evaluation for the Brumley-Boneh real oracle.

The 2026-09-11 attribution probe showed the `real` noise is one-sided
interrupt/scheduling contamination (NOT frequency drift), and that a MINIMUM
estimator removes ~99.9% of it (min-of-20 -> ~0.08% rel std).  The victim
already supports this via `repeat` (real-mode measure() takes min over repeat),
but every prior run used repeat=1.  This runner engages repeat>>1 and:

  1. reads gap-separability AUC with the min-filtered oracle, UNPINNED then PINNED
     (to see whether pinning even matters once the interrupts are min-filtered);
  2. if AUC clears the threshold, launches a full pinned real recovery
     (repeat-min + beam) finishing with Coppersmith.

Run:  python run_min_estimator_eval.py   (via ..\\venv\\Scripts\\python.exe)
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
from bb_victim import load_crt_key, generate_crt_key, CRTRSAVictim  # noqa: E402

# ------------------------------- config ------------------------------------ #
KEY_BITS = 256
BATCH = 32
REPEAT = 20                # min over this many batch-measures = the min estimator

DIAG_TOP_BITS = 16
DIAG_NEIGH = 25
DIAG_SAMPLE = 1
DIAG_TRIALS = 8

AUC_GO = 0.90              # min-filtered oracle should be ~1.0

# full recovery (min estimator, cache-friendly: interleave off since drift is
# no longer the issue -> beam can reuse the cross-depth measurement cache)
REC_NEIGH = 64
REC_SAMPLE = 1
REC_BEAM = 8
REC_TAIL_BRUTE = 48       # Coppersmith finish on the low 48 bits
# --------------------------------------------------------------------------- #


def load_key():
    cache = os.path.join(_ABB, "bb_cache", "keybits_%d_random" % KEY_BITS,
                         "key.json")
    return load_crt_key(cache) if os.path.exists(cache) else generate_crt_key(KEY_BITS)


def run_diag(tag):
    key = load_key()
    vic = CRTRSAVictim(key, timing="real", batch=BATCH, repeat=REPEAT)
    R = vic.q_radix()
    rinv = pow(R, -1, vic.n)
    qbits = vic.q.bit_length()
    t0 = time.perf_counter()
    diag = abb.diagnose_gap_separability(
        vic, rinv, qbits, top_bits=DIAG_TOP_BITS, neigh=DIAG_NEIGH,
        sample=DIAG_SAMPLE, trials=DIAG_TRIALS, verbose=True)
    dt = time.perf_counter() - t0
    print("[%s] block AUC=%.3f  interleave AUC=%.3f  (%.0fs, %d queries)"
          % (tag, diag["block"]["auc"], diag["interleave"]["auc"], dt,
             vic.queries))
    return diag


def main():
    print("########## min-estimator diagnostic — UNPINNED ##########")
    diag_unpin = run_diag("unpinned")

    print("\n########## min-estimator diagnostic — PINNED ##########")
    cpu_pin.pin_process(core=0, high_priority=True, verbose=True)
    cpu_pin.warmup(1.5)
    diag_pin = run_diag("pinned")

    best = max(diag_unpin["block"]["auc"], diag_unpin["interleave"]["auc"],
               diag_pin["block"]["auc"], diag_pin["interleave"]["auc"])
    print("\n[summary] best AUC across {unpinned,pinned}x{block,interleave} = %.3f"
          % best, flush=True)
    with open(os.path.join(_ABB, "min_estimator_diag.json"), "w") as fh:
        json.dump({"unpinned": diag_unpin, "pinned": diag_pin,
                   "best_auc": best}, fh, indent=2)

    if "diag-only" in sys.argv:
        print("[diag-only] skipping recovery launch.", flush=True)
        return

    if best < AUC_GO:
        print("[stop] AUC below %.2f even with the min estimator; not launching "
              "the long recovery. See report." % AUC_GO, flush=True)
        return

    print("[go] AUC clears %.2f -> launching FULL pinned min-estimator recovery "
          "(repeat=%d, beam=%d) + Coppersmith." % (AUC_GO, REPEAT, REC_BEAM),
          flush=True)
    abb.run(key_bits=KEY_BITS, timing="real", neigh=REC_NEIGH,
            sample=REC_SAMPLE, batch=BATCH, repeat=REPEAT,
            recover_bits=None, tail_brute=REC_TAIL_BRUTE,
            beam_width=REC_BEAM, interleave=False, pin_cpu=True, pin_core=0,
            warmup_s=0.0, make_plots=True, use_key_cache=True, verbose=True)


if __name__ == "__main__":
    main()
