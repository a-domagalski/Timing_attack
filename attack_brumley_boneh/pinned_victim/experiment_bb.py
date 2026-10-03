"""
Reproducible driver for the Brumley-Boneh CRT-RSA timing attack.

Runs two demonstrations and leaves all figures + JSON on disk under
attack_brumley_boneh/bb_results/:

  1. EXACT oracle (noise-free extra-reduction counts): full binary-search
     recovery of the factor q and complete factorization of N -> the private
     key.  This is the deterministic demonstrator (software analogue of Dhem's
     fixed-clock smartcard), and it produces the characterization figures:
        Fig 1  - extra reductions vs g (the Schindler sawtooth; drops at q, p)
        Fig 2b - zero-one gap vs neighborhood size (0-bit grows, 1-bit flat)
        Fig 3  - zero-one gap per recovered bit (0-bits above the axis)
        Fig 4  - histogram of the gap by true bit value (separability)

  2. REAL oracle (__rdtsc wall-clock seconds): a smaller run recovering the top
     bits, to show the same zero-one-gap structure under genuine timing noise
     and to produce Fig 2a (the sample-size convergence).

Run (from the bundle root, after building the DLL - see attack_refined/montmul_c.py):
    venv\\Scripts\\python.exe attack_brumley_boneh\\experiment_bb.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABB = os.path.dirname(_HERE)                 # attack_brumley_boneh/ (shared modules + bb_results)
_ROOT = os.path.dirname(_ABB)                 # montMul/
for _p in (_HERE, _ABB, os.path.join(_ROOT, "attack_refined"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import attack_brumley_boneh as abb

# ------------------------------------------------------------------ config -- #
EXACT_KEY_BITS = 256      # exact demo: modulus size (q ~ 128 bits)
EXACT_NEIGH = 64          # neighborhood for the exact recovery
EXACT_TAIL_BRUTE = 16     # brute-force the lowest 16 bits to finish (Coppersmith stand-in)
EXACT_BEAM = 1            # exact is noise-free -> greedy already recovers all of q

REAL_KEY_BITS = 256       # real demo: modulus size
REAL_NEIGH = 100          # neighborhood for the (noisy) real recovery
REAL_SAMPLE = 3           # median over this many measures per value
REAL_BATCH = 40           # rdtsc-timed ladders averaged per half
REAL_RECOVER_BITS = 20    # recover just the top bits (keeps the timed run short)
REAL_BEAM = 1             # >1 enables confidence beam / backtracking on the noisy
                          #    oracle (fixes marginal-bit propagation; ~x the
                          #    queries, so raise REAL_RECOVER_BITS + REAL_TAIL to
                          #    <=26 for a deep run that can actually factor N).
REAL_TAIL_BRUTE = 16      # low bits to brute-force at the end of a real run
REAL_INTERLEAVE = False   # measure g/ghi back-to-back so CPU-frequency drift
                          #    cancels in the zero-one gap (A/B vs block mode)
# ---------------------------------------------------------------------------- #


def main():
    print("\n########## 1) EXACT ORACLE - full factorization demo ##########")
    exact = abb.run(
        key_bits=EXACT_KEY_BITS, timing="exact", neigh=EXACT_NEIGH,
        sample=1, tail_brute=EXACT_TAIL_BRUTE, beam_width=EXACT_BEAM,
        make_plots=True, use_key_cache=True, verbose=True,
    )

    print("\n########## 2) REAL ORACLE - top-bit recovery under noise ##########")
    real = abb.run(
        key_bits=REAL_KEY_BITS, timing="real", neigh=REAL_NEIGH,
        sample=REAL_SAMPLE, batch=REAL_BATCH, repeat=1,
        recover_bits=REAL_RECOVER_BITS, tail_brute=REAL_TAIL_BRUTE,
        beam_width=REAL_BEAM, interleave=REAL_INTERLEAVE, make_plots=True,
        use_key_cache=True, verbose=True,
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT SUMMARY")
    print("=" * 78)
    print("  exact : %d-bit N | q recovered & N factored = %s | d matches = %s"
          % (exact["modulus_bits"], exact["factored"],
             exact["d_recovered_matches"]))
    print("          per-bit accuracy %.1f%% over %d gap bits, %d queries"
          % (100 * exact["per_bit_accuracy"], exact["bits_recovered_via_gap"],
             exact["queries"]))
    print("  real  : %d-bit N | top-bit gap accuracy %.1f%% over %d bits, %d queries"
          % (real["modulus_bits"], 100 * real["per_bit_accuracy"],
             real["bits_recovered_via_gap"], real["queries"]))
    print("  artifacts under: %s"
          % os.path.join(_ABB, "bb_results"))
    print("=" * 78)


if __name__ == "__main__":
    main()
