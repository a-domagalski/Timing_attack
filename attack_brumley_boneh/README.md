# Brumley-Boneh CRT-RSA timing attack

Recovers the secret prime factor `q` of `N = p*q` by timing CRT-RSA decryption
(Schindler's Montgomery extra-reduction channel), then factors `N` and derives
`d`. Reproduces D. Brumley & D. Boneh, *"Remote Timing Attacks are Practical"*
(USENIX 2003).

There are **two attack versions**, distinguished by whether the victim's CPU
clock is controlled by the attacker:

| version | victim | timing source | dir |
|---|---|---|---|
| **pinned victim** | in-process (same process as attacker, pinned core) | `__rdtsc` cycles measured *inside* C | `pinned_victim/` |
| **unpinned victim** | separate OS process, NOT pinned (realistic server) | attacker-measured socket round-trip (wall clock) | `unpinned_victim/` |

The pinned version is the noise-free / fixed-clock demonstrator (per-bit oracle
AUC 1.0, full key recovered). The unpinned version is the faithful remote
setting; its own turbo/DVFS frequency drift originally floored it at AUC ~0.75.
See `unpinned_victim/` for the fix.

## Layout

```
attack_brumley_boneh/
├── attack_brumley_boneh.py   # SHARED CORE: in-process attacker + the recovery
│                             #   machinery reused by BOTH versions
│                             #   (neighborhood_gap, recover_factor_beam,
│                             #    complete_factor, diagnose_gap_separability).
│                             #   Run directly = the in-process attack.
├── bb_victim.py              # SHARED: in-process CRT-RSA victim + key gen/load
│                             #   utilities (used by both versions).
├── coppersmith.py            # SHARED: Coppersmith top-bits-known completion.
├── cpu_pin.py                # SHARED: CPU affinity / priority / warm-up helper.
│
├── pinned_victim/            # VERSION 1 — in-process, pinned/exact/real oracle
│   ├── experiment_bb.py            # reproducible exact + small real driver (figs+JSON)
│   ├── run_pinned_eval.py          # pinned real-oracle AUC diag + full recovery
│   ├── run_min_estimator_eval.py   # min-estimator (repeat>>1) eval, unpinned vs pinned
│   └── attribution_probe.py        # noise-attribution probe (freq drift vs interrupts)
│
├── unpinned_victim/          # VERSION 2 — remote socket victim, wall-clock timed
│   ├── attack_remote.py            # THE ATTACK: smoke | diag | partial | full
│   ├── bb_victim_server.py         # standalone unpinned victim (localhost TCP)
│   └── remote_probe.py             # channel characterisation probe (noise + per-bit AUC)
│
├── bb_cache/                 # shared: cached target keys (per key size / source)
└── bb_results/               # shared: run artifacts, grouped by victim model + regime
    ├── inprocess_victim/        # exact_oracle/ real_unpinned/ real_pinned/ attribution/
    └── remote_unpinned_victim/  # remote_*, remote_probe_*
```

Shared modules stay at the top level because both versions import them
(`import attack_brumley_boneh`, `cpu_pin`, `coppersmith`, `from bb_victim import ...`).
`bb_cache/` and `bb_results/` are shared output dirs; every script writes there
regardless of which subdirectory it lives in.

## Running (from the `side-channel-attack-montMul` root, using its venv)

```powershell
# --- pinned victim (in-process) ---
venv\Scripts\python.exe attack_brumley_boneh\attack_brumley_boneh.py         # in-process attack (edit __main__ CONFIG)
venv\Scripts\python.exe attack_brumley_boneh\pinned_victim\experiment_bb.py  # exact + real demo
venv\Scripts\python.exe attack_brumley_boneh\pinned_victim\attribution_probe.py

# --- unpinned victim (remote wall clock) ---
venv\Scripts\python.exe attack_brumley_boneh\unpinned_victim\attack_remote.py smoke    # correctness + timing sanity
venv\Scripts\python.exe attack_brumley_boneh\unpinned_victim\attack_remote.py diag     # block vs interleave AUC
venv\Scripts\python.exe attack_brumley_boneh\unpinned_victim\attack_remote.py partial  # bounded top-bit recovery
venv\Scripts\python.exe attack_brumley_boneh\unpinned_victim\attack_remote.py full     # full key (multi-hour)
venv\Scripts\python.exe attack_brumley_boneh\unpinned_victim\remote_probe.py           # channel probe
```

Requires the C Montgomery core `attack_refined/cext/montmul.dll` (see
`attack_refined/montmul_c.py` for the build command).

## The unpinned-victim fix (interleaved-min pairing)

The unpinned victim's turbo/DVFS drift moves the true compute-time floor itself,
so a per-cipher `min` (which only removes the one-sided interrupt tail) cannot
remove it: measuring the `g` and `ghi` neighbourhoods in separate windows lets
the drift corrupt the zero-one gap `T(g) - T(ghi)`. The fix (`INTERLEAVE=True`
→ `RemoteVictim.measure_pair`) measures `g,ghi,g,ghi,…` at the round-trip level
and takes the min of each side from the **same** window, so the drift is
common-mode and cancels. Measured top-bit gap AUC:

| estimator | AUC |
|---|---|
| block (old) | 0.62–0.81 |
| interleaved-min, repeat=12 | 0.79 |
| interleaved-min, repeat=24, neigh=96 | 0.98 |
| interleaved-min, repeat=48, neigh=96 | **1.000** (matches pinned) |

The `remote_probe.py` estimator sweep (block / paired / paired_minmin /
paired_avg) is what established this.
