# Learning-based per-bit distinguisher (Pillars A & B)

ML arm of the Brumley-Boneh unpinned-victim timing attack: a learned, calibrated
per-bit distinguisher meant to replace/augment the classical `sign(delta)` bit
decision that feeds the beam recovery. Implements two of the proposed pillars
(work-log 2026-09-29).

## Files

| file | role |
|---|---|
| `collect_timings_dataset.py` | collects the training DB, reusing the SAME measurement path as the successful attack (victim `bb_victim_server.py`, `RemoteVictim`, `neighborhood_gap`). Records per (key,bit): `MEAS_PER_BIT` measurements `{Tg,Tghi,delta,ref,elapsed_s}`, the true bit label, the noise-free extra-reduction gap `exact.ex_delta` (Pillar A teacher), and a fixed reference round-trip `ref` (Pillar B drift covariate). Append-only JSONL, one dir per param set. |
| `train_distinguisher.py` | feature engineering + models + grouped-by-key CV + calibration metrics + sweeps. |
| `collect_kb128_weak.json` | collection config for the tested regime. |

## What Pillars A and B are (as implemented)

- **Pillar A — privileged-information / LUPI distillation.** A teacher MLP is
  trained on the *privileged* noise-free `ex_delta` (available only at training
  time, computed on a clone) and the student distils its soft targets
  (temperature τ); a student **aux head regresses `ex_delta`**. The aux head
  needs a *shared representation* to influence the bit head, so the pillar
  configs use a small **shared linear bottleneck** (with a purely linear student
  the aux head is inert — see findings).
- **Pillar B — cross-time drift adaptation.** A gradient-reversal (DANN) domain
  classifier predicts a discrete drift bin (from the reference `ref` level) from
  the shared representation, pushing it to be drift-invariant. Plus **per-key
  feature normalization** (z-score the absolute-level features within each key)
  so features transfer across keys/victim-processes.

Everything the student uses at inference is attack-observable (timings only);
`ex_delta` and the true bit are training-only.

## Usage (from the montMul root, its venv)

```
venv\Scripts\python.exe attack_brumley_boneh\unpinned_victim\attack_nn\collect_timings_dataset.py --config attack_brumley_boneh\unpinned_victim\attack_nn\collect_kb128_weak.json --keys 12
venv\Scripts\python.exe attack_brumley_boneh\unpinned_victim\attack_nn\train_distinguisher.py <nn_datasets\...dir> --folds 5 --epochs 250
venv\Scripts\python.exe ...\train_distinguisher.py <dir> --meas-sweep 1,2,4,8      # #measurements axis
venv\Scripts\python.exe ...\train_distinguisher.py <dir> --sweep                    # distill/aux/dann axis
venv\Scripts\python.exe ...\train_distinguisher.py <dir> --no-perkey                # ablate per-key norm
```
Collection is append-only, so re-running grows the dataset. `--max-meas N` uses
only the first N measurements per bit.

## Metrics

Grouped 5-fold CV **by key** (train on some keys, test on held-out keys — the
cross-key generalization the profiled attack needs). Reports accuracy, AUC, ECE
(calibration), NLL. Calibration matters because the beam consumes per-bit
*confidences*, not just hard decisions.

## Findings (kb128, q=64; block estimator; batch=40, repeat=12, neigh=16; 32 keys / 384 bits)

Smallest workable key = **128-bit** (q=64): 64/96-bit keys give a zero
extra-reduction gap (no signal). On this quiet localhost the **block** estimator
separates far better than the interleaved-min one (opposite of the real drifting
host, where interleave was the fix) — localhost has little DVFS drift.

1. **A calibrated linear learned distinguisher matches the classical rule on
   accuracy/AUC and clearly improves calibration**, consistently across
   measurement counts 1–8:
   - meas=1: `logreg` AUC 0.595 / **ECE 0.089** / NLL 0.678 vs `sign(delta)` AUC
     0.566 / ECE 0.175 / NLL 0.760.
   - meas=8: `logreg` AUC 0.611 / **ECE 0.129** vs baseline 0.616 / 0.160.
   This is the beam-relevant win (better-calibrated confidences).
2. **Pillar A (privileged distillation) gives no accuracy/AUC gain here.** The
   transferable signal is essentially 1-D (the sign of `delta`), so the
   privileged teacher cannot inject extra separability; increasing the
   distillation weight λ monotonically *hurts* (AUC 0.611→0.571 as λ:0→0.75).
   Diagnosed + fixed an implementation subtlety: with a *pure* linear student
   the aux head is inert (shares no representation) — the shared linear
   bottleneck makes it functional, but it still doesn't help on this 1-D signal.
3. **Pillar B (drift adaptation) has no effect on this drift-free localhost** —
   every DANN weight gives identical results. Expected: there is no turbo/DVFS
   drift here to remove. It targets the real host's drift (the documented derail
   cause), which localhost lacks.

### Where the pillars should matter (next experiments)
- **Pillar B:** collect on the real, drifting unpinned host (or a busy machine),
  where `ref`/level carries exploitable drift — the regime that caused the
  derail lottery.
- **Pillar A:** larger/richer-signal keys (more limbs → multi-dimensional
  leakage beyond a single gap sign) and more keys, with a shared encoder, so the
  privileged `ex_delta` margin can shape a non-trivial representation.
- Integrate the calibrated confidence into `recover_factor_beam` (replace the
  `delta`/`tau` decision) and measure whole-key recovery, not just per-bit AUC.

All raw numbers are in the dataset dir: `eval_report.json`, `eval_sweep.json`,
`eval_meas_sweep.json`.
