"""
Per-bit distinguisher for the Brumley-Boneh unpinned-victim timing channel,
implementing Pillar A (privileged-information / LUPI distillation) and Pillar B
(cross-time drift adaptation), trained on the datasets produced by
``collect_timings_dataset.py``.

Context (see work-log 2026-09-29)
---------------------------------
The live attack decides each bit of q by thresholding the zero-one gap
``delta = T(g) - T(ghi)`` (classical Brumley-Boneh).  The documented failure
mode is the drift-driven "derail lottery": a single marginal/contaminated bit
flips and the beam loses the true path.  This module trains a *calibrated*
per-bit distinguisher meant to (a) be better calibrated than a raw threshold
(so the beam keeps marginal-but-correct bits) and (b) be robust to the victim's
turbo/DVFS drift.

Pillar A — privileged-information distillation (Vapnik LUPI / Lopez-Paz
generalized distillation).  At *training* time only we have the NOISE-FREE
Montgomery extra-reduction gap ``ex_delta`` (computed on a clone we control).
It is NOT available to the live attacker, so it is "privileged".  We use it two
ways: (1) a TEACHER MLP trained on the privileged feature produces soft targets
the student distills; (2) an AUXILIARY head on the student regresses the
(standardised) ``ex_delta``.  The magnitude of ``ex_delta`` tells the student
which bits were *intrinsically marginal* (small true gap) vs easy -- exactly the
information a hard 0/1 label hides and the derail cares about.

Pillar B — cross-time drift adaptation (domain-adversarial, Ganin GRL).  The
victim's frequency drift is the nuisance "domain".  We derive a discrete drift
label from the key-independent reference round-trip ``ref`` (or, if absent, the
absolute timing level ``Tg+Tghi``) and attach a gradient-reversed domain
classifier, so the shared representation is pushed to be drift-invariant while
still predicting the bit.  This is the cross-TIME analogue of the published
cross-DEVICE domain adaptation.

Everything the student uses at inference is attack-observable (timings only);
the privileged ``ex_delta`` and the true bit are used only during training.

Run (from the montMul root, using its venv)::

    venv\\Scripts\\python.exe attack_brumley_boneh\\unpinned_victim\\attack_nn\\train_distinguisher.py <dataset_dir>
    venv\\Scripts\\python.exe ... train_distinguisher.py <dir> --sweep      # hyperparameter sweep
    venv\\Scripts\\python.exe ... train_distinguisher.py <dir> --folds 5 --epochs 200

Outputs a printed table + a JSON report (``eval_report.json``) in the dataset dir.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import collect_timings_dataset as cds  # noqa: E402

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:                       # pragma: no cover
    _HAS_TORCH = False


# =========================================================================== #
# Feature engineering: aggregate the MEAS_PER_BIT measurements into ONE per-bit
# feature vector (the attack decides per bit using all its measurements).
# =========================================================================== #
_REGULAR_FEATURES = [
    "delta_mean", "delta_min", "delta_max", "delta_std", "delta_median",
    "vote_pos", "Tg_mean", "Tghi_mean", "level_mean",
    "delta_over_level", "ref_mean", "ref_std", "delta_over_ref",
    "log_neigh", "bitpos_rel",
]


def _safe_div(a, b):
    return a / b if b not in (0, 0.0) else 0.0


def build_perbit_table(records, max_meas=None):
    """Group the flattened per-measurement samples back to per-bit rows and
    compute a fixed feature vector per bit.

    ``max_meas``: if set, use only the first ``max_meas`` measurements per bit
    (the "number of measurements per bit" parameter axis) -- lets us study how
    the baseline and the Pillar models improve with more measurements WITHOUT
    re-collecting.

    Returns a dict of numpy arrays:
      X       (N, F)  regular (attack-observable) features
      y       (N,)    true bit
      xp      (N, P)  privileged features (ex_delta-derived); NaN if unavailable
      drift   (N,)    continuous drift proxy (ref level, else timing level)
      groups  (N,)    key_index (for grouped CV)
      has_priv (bool) whether privileged features are present for all rows
    """
    flat = cds.flatten_per_bit(records, include_exact=True)
    # group by (key_index, bit_pos)
    buckets = {}
    for s in flat:
        key = (s["key_index"], s["bit_pos"])
        buckets.setdefault(key, []).append(s)

    max_bitpos = max((s["bit_pos"] for s in flat), default=1) or 1

    X, y, xp, drift, groups = [], [], [], [], []
    priv_ok = True
    for (ki, bp), ms in buckets.items():
        if max_meas is not None:
            ms = ms[:max_meas]
        deltas = np.array([m["features"]["delta"] for m in ms], dtype=float)
        Tg = np.array([m["features"]["Tg"] for m in ms], dtype=float)
        Tghi = np.array([m["features"]["Tghi"] for m in ms], dtype=float)
        neigh = float(ms[0]["features"]["neigh"] or 1)
        refs = np.array([(m["ref"] if m.get("ref") is not None else np.nan)
                         for m in ms], dtype=float)
        has_ref = not np.all(np.isnan(refs))

        delta_mean = float(np.mean(deltas))
        Tg_mean = float(np.mean(Tg))
        Tghi_mean = float(np.mean(Tghi))
        level_mean = Tg_mean + Tghi_mean
        ref_mean = float(np.nanmean(refs)) if has_ref else level_mean
        ref_std = float(np.nanstd(refs)) if has_ref else 0.0

        feat = {
            "delta_mean": delta_mean,
            "delta_min": float(np.min(deltas)),
            "delta_max": float(np.max(deltas)),
            "delta_std": float(np.std(deltas)),
            "delta_median": float(np.median(deltas)),
            "vote_pos": float(np.mean(deltas > 0)),
            "Tg_mean": Tg_mean,
            "Tghi_mean": Tghi_mean,
            "level_mean": level_mean,
            "delta_over_level": _safe_div(delta_mean, level_mean),
            "ref_mean": ref_mean,
            "ref_std": ref_std,
            "delta_over_ref": _safe_div(delta_mean, ref_mean),
            "log_neigh": math.log(neigh + 1.0),
            "bitpos_rel": bp / max_bitpos,
        }
        X.append([feat[k] for k in _REGULAR_FEATURES])
        y.append(int(ms[0]["label"]))

        exd = ms[0].get("exact_delta")
        if exd is None:
            priv_ok = False
            xp.append([np.nan, np.nan, np.nan])
        else:
            exd = float(exd)
            xp.append([exd, math.copysign(1.0, exd) if exd != 0 else 0.0,
                       math.log1p(abs(exd))])

        drift.append(ref_mean if has_ref else level_mean)
        groups.append(int(ki))

    return {
        "X": np.array(X, dtype=np.float32),
        "y": np.array(y, dtype=np.int64),
        "xp": np.array(xp, dtype=np.float32),
        "drift": np.array(drift, dtype=np.float32),
        "groups": np.array(groups, dtype=np.int64),
        "has_priv": priv_ok,
        "feature_names": list(_REGULAR_FEATURES),
    }


# feature groups for per-key normalization (cross-key transfer)
_DELTA_COLS = ("delta_mean", "delta_min", "delta_max", "delta_median",
               "delta_std")          # scale by key delta-scale, NO centering
                                     # (centering would destroy the sign that
                                     #  carries the bit)
_LEVEL_COLS = ("Tg_mean", "Tghi_mean", "level_mean", "ref_mean", "ref_std")
#              z-score within key: removes the per-key/per-process absolute
#              offset (pure nuisance across keys) while KEEPING the within-key
#              drift variation -- the signal Pillar B is meant to exploit.


def perkey_normalize(data):
    """Normalize timing features WITHIN each key so they transfer across keys.

    The raw absolute timings (Tg, level, ref) carry a per-key/per-victim-process
    offset that does NOT transfer to a held-out key, so a model using them
    overfits the offset (this is why raw-feature MLPs lost to sign(delta)).  The
    attacker legitimately has all of a target key's own measurements, so this
    normalization is attack-realisable (use a running estimate online).

    Delta-type columns are divided by the key's delta-scale (no centering, to
    preserve the sign that encodes the bit); level-type columns are z-scored
    within the key.  Returns a NEW data dict (does not mutate the input).
    """
    X = data["X"].copy()
    groups = data["groups"]
    names = data["feature_names"]
    di = [names.index(c) for c in _DELTA_COLS if c in names]
    li = [names.index(c) for c in _LEVEL_COLS if c in names]
    delta_mean_i = names.index("delta_mean")
    for k in np.unique(groups):
        m = groups == k
        # delta scale from the key's own delta_mean spread (sign preserved)
        sd = np.std(X[m, delta_mean_i]) + 1e-12
        for c in di:
            X[m, c] = X[m, c] / sd
        for c in li:
            col = X[m, c]
            X[m, c] = (col - np.mean(col)) / (np.std(col) + 1e-12)
    out = dict(data)
    out["X"] = X
    return out


# =========================================================================== #
# Metrics (no sklearn).
# =========================================================================== #
def auc_score(y, p):
    """Mann-Whitney AUC via rank averaging."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    sum_pos = np.sum(ranks[y == 1])
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece_score(y, p, bins=10):
    """Expected calibration error."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    n = len(p)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        conf = np.mean(p[m])
        acc = np.mean(y[m])
        e += (np.sum(m) / n) * abs(acc - conf)
    return float(e)


def nll_score(y, p, eps=1e-7):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def all_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {
        "acc": float(np.mean(yhat == y)),
        "auc": auc_score(y, p),
        "ece": ece_score(y, p),
        "nll": nll_score(y, p),
    }


# =========================================================================== #
# Baselines (no NN).
# =========================================================================== #
def baseline_sign(Xtr, ytr, Xte, yte, feat_names):
    """Classical rule: threshold the mean zero-one gap at 0, pick the sign
    direction that maximises training accuracy."""
    di = feat_names.index("delta_mean")
    dtr, dte = Xtr[:, di], Xte[:, di]
    # direction: bit = 1 if delta<0 (dir=-1) or delta>0 (dir=+1)
    best_dir, best_acc = 1, -1
    for d in (1, -1):
        pred = ((d * dtr) < 0).astype(int)  # bit=1 when d*delta<0
        acc = np.mean(pred == ytr)
        if acc > best_acc:
            best_acc, best_dir = acc, d
    # map to a pseudo-probability via a logistic on the (scaled) gap for AUC/ECE
    scale = np.std(dtr) + 1e-9
    p = 1.0 / (1.0 + np.exp(best_dir * dte / scale))  # higher when d*delta<0
    return all_metrics(yte, p)


# =========================================================================== #
# Torch model: shared encoder + bit head + aux (privileged) head + domain head.
# =========================================================================== #
if _HAS_TORCH:
    class _GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambd):
            ctx.lambd = lambd
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lambd * g, None

    def grad_reverse(x, lambd):
        return _GRL.apply(x, lambd)

    class Distinguisher(nn.Module):
        def __init__(self, in_dim, hid=16, n_domains=4, bottleneck=0):
            super().__init__()
            if bottleneck and bottleneck > 0:
                # SHARED LINEAR bottleneck: keeps the bit predictor (near-)linear
                # but forces the bit/aux/domain heads through ONE shared
                # representation -- so Pillar A's aux (ex_delta regression) and
                # Pillar B's adversarial domain loss actually shape the features
                # the bit head uses (with a purely linear student they would be
                # inert, sharing nothing).
                self.enc = nn.Linear(in_dim, bottleneck)
                rep = bottleneck
            elif hid and hid > 0:
                self.enc = nn.Sequential(
                    nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hid, hid), nn.ReLU(),
                )
                rep = hid
            else:
                self.enc = nn.Identity()
                rep = in_dim
            self.bit_head = nn.Linear(rep, 1)
            self.aux_head = nn.Linear(rep, 1)          # regress ex_delta (std)
            self.dom_head = nn.Sequential(
                nn.Linear(rep, max(8, rep)), nn.ReLU(),
                nn.Linear(max(8, rep), n_domains))

        def forward(self, x, grl_lambda=0.0):
            h = self.enc(x)
            return (self.bit_head(h).squeeze(-1),
                    self.aux_head(h).squeeze(-1),
                    self.dom_head(grad_reverse(h, grl_lambda)))

    class Teacher(nn.Module):
        """Privileged teacher for distillation (trained on ex_delta features)."""
        def __init__(self, in_dim, hid=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)


def _standardize(train, *others):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd[sd < 1e-9] = 1.0
    out = [(train - mu) / sd]
    for o in others:
        out.append((o - mu) / sd)
    return out + [mu, sd]


def _drift_bins(drift_tr, drift_te, k=4):
    """Quantile-bin the drift proxy into k domains (edges from train)."""
    qs = np.quantile(drift_tr, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(drift_tr, qs), np.digitize(drift_te, qs)


def train_eval_fold(data, tr, te, cfg, seed=0):
    """Train the configured student on fold `tr`, evaluate on `te`.

    cfg keys: hid, epochs, lr, wd, distill (λ), temp (τ), aux (β), dann (α),
              n_domains.
    Returns metrics dict for the student.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, xp, drift = data["X"], data["y"], data["xp"], data["drift"]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    Xtr_s, Xte_s, _, _ = _standardize(Xtr, Xte)
    in_dim = Xtr_s.shape[1]

    use_priv = data["has_priv"] and (cfg["distill"] > 0 or cfg["aux"] > 0)
    if use_priv:
        Xp_tr, Xp_te, _, _ = _standardize(xp[tr], xp[te])
        # standardized ex_delta regression target (first privileged column)
        exd_tr = xp[tr][:, 0]
        exd_mu, exd_sd = np.mean(exd_tr), np.std(exd_tr) + 1e-9
        exd_tr_s = (exd_tr - exd_mu) / exd_sd

    dom_tr, dom_te = _drift_bins(drift[tr], drift[te], cfg["n_domains"])

    dev = "cpu"
    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=dev)
    dom_tr_t = torch.tensor(dom_tr, dtype=torch.long, device=dev)

    # ---- optional privileged teacher (generalized distillation) ---- #
    teacher_soft = None
    if use_priv and cfg["distill"] > 0:
        Tnet = Teacher(Xp_tr.shape[1]).to(dev)
        Xp_tr_t = torch.tensor(Xp_tr, dtype=torch.float32, device=dev)
        opt_t = torch.optim.Adam(Tnet.parameters(), lr=1e-2, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        Tnet.train()
        for _ in range(200):
            opt_t.zero_grad()
            logit = Tnet(Xp_tr_t)
            loss = bce(logit, ytr_t)
            loss.backward()
            opt_t.step()
        Tnet.eval()
        with torch.no_grad():
            teacher_soft = torch.sigmoid(Tnet(Xp_tr_t) / cfg["temp"])

    # ---- student ---- #
    net = Distinguisher(in_dim, hid=cfg["hid"], n_domains=cfg["n_domains"],
                        bottleneck=cfg.get("bottleneck", 0)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["wd"])
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    if use_priv:
        exd_tr_t = torch.tensor(exd_tr_s, dtype=torch.float32, device=dev)

    epochs = cfg["epochs"]
    net.train()
    for ep in range(epochs):
        # GRL schedule (Ganin): ramp lambda 0->dann
        p = ep / max(1, epochs - 1)
        grl = cfg["dann"] * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)
        opt.zero_grad()
        bit_logit, aux, dom_logit = net(Xtr_t, grl_lambda=grl)
        loss = (1.0 - cfg["distill"]) * bce(bit_logit, ytr_t)
        if teacher_soft is not None:
            # distillation: match soft targets (temperature-scaled)
            s = torch.sigmoid(bit_logit / cfg["temp"])
            kd = nn.functional.binary_cross_entropy(s, teacher_soft.detach())
            loss = loss + cfg["distill"] * (cfg["temp"] ** 2) * kd
        if use_priv and cfg["aux"] > 0:
            loss = loss + cfg["aux"] * mse(aux, exd_tr_t)
        if cfg["dann"] > 0:
            loss = loss + ce(dom_logit, dom_tr_t)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        bit_logit, _, _ = net(Xte_t, grl_lambda=0.0)
        p_te = torch.sigmoid(bit_logit).cpu().numpy()
    return all_metrics(yte, p_te)


# =========================================================================== #
# Grouped cross-validation driver.
# =========================================================================== #
def grouped_folds(groups, k, seed=0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, k)
    folds = []
    for i in range(k):
        te_keys = set(chunks[i].tolist())
        te = np.array([j for j, g in enumerate(groups) if g in te_keys])
        tr = np.array([j for j, g in enumerate(groups) if g not in te_keys])
        if len(te) and len(tr):
            folds.append((tr, te))
    return folds


def _mean_std(vals):
    a = np.array([v for v in vals if not math.isnan(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a))


def run_models(data, folds, cfgs, seed=0):
    """Evaluate the baseline + each configured model across folds."""
    results = {}
    # baseline (no NN)
    base = {"acc": [], "auc": [], "ece": [], "nll": []}
    for tr, te in folds:
        Xtr_s, Xte_s, _, _ = _standardize(data["X"][tr], data["X"][te])
        m = baseline_sign(Xtr_s, data["y"][tr], Xte_s, data["y"][te],
                          data["feature_names"])
        for k in base:
            base[k].append(m[k])
    results["baseline_sign"] = {k: _mean_std(v) for k, v in base.items()}

    if not _HAS_TORCH:
        return results

    for name, cfg in cfgs.items():
        acc = {"acc": [], "auc": [], "ece": [], "nll": []}
        for fi, (tr, te) in enumerate(folds):
            # average over a few seeds to damp small-data init variance
            per = {"acc": [], "auc": [], "ece": [], "nll": []}
            for sd in range(3):
                m = train_eval_fold(data, tr, te, cfg, seed=seed + 100 * sd + fi)
                for k in per:
                    per[k].append(m[k])
            for k in acc:
                acc[k].append(float(np.nanmean(per[k])))
        results[name] = {k: _mean_std(v) for k, v in acc.items()}
    return results


def _base_cfg(**over):
    cfg = {"hid": 16, "epochs": 200, "lr": 5e-3, "wd": 1e-3,
           "distill": 0.0, "temp": 2.0, "aux": 0.0, "dann": 0.0,
           "n_domains": 4, "bottleneck": 0}
    cfg.update(over)
    return cfg


def default_model_suite():
    # logreg_plain: purely-linear student (calibrated logistic regression) --
    #   the strong, non-overfitting backbone for this ~1-D signal.
    # pillar* : add a SHARED LINEAR bottleneck so Pillar A's aux (ex_delta) and
    #   Pillar B's adversarial domain loss actually shape the bit head's
    #   representation (they are inert with a pure-linear student).
    return {
        "logreg_plain":  _base_cfg(hid=0),
        "pillarA_lupi":  _base_cfg(hid=0, bottleneck=6, distill=0.5, aux=0.5),
        "pillarAB_full": _base_cfg(hid=0, bottleneck=6, distill=0.5, aux=0.5,
                                   dann=0.3),
        "mlp_plain":     _base_cfg(hid=16),
    }


def sweep_suite():
    suite = {}
    for lam in (0.0, 0.25, 0.5, 0.75):
        suite["A_distill%.2f" % lam] = _base_cfg(hid=0, distill=lam)
    for beta in (0.25, 0.5, 1.0):
        suite["A_aux%.2f" % beta] = _base_cfg(hid=0, aux=beta)
    for alpha in (0.1, 0.3, 0.5):
        suite["B_dann%.2f" % alpha] = _base_cfg(hid=0, distill=0.5, aux=0.5,
                                                dann=alpha)
    return suite


def _print_table(results, title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    print("%-18s %14s %14s %12s %12s" % ("model", "acc", "auc", "ece", "nll"))
    print("-" * 74)
    for name, m in results.items():
        def fmt(k):
            mu, sd = m[k]
            return "%.3f+-%.3f" % (mu, sd)
        print("%-18s %14s %14s %12s %12s"
              % (name, fmt("acc"), fmt("auc"), fmt("ece"), fmt("nll")))


def main():
    ap = argparse.ArgumentParser(description="Pillar A/B per-bit distinguisher.")
    ap.add_argument("dataset_dir", help="a nn_datasets/<param_sig> directory")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="run the hyperparameter sweep instead of the suite")
    ap.add_argument("--max-meas", type=int, default=None,
                    help="use only the first N measurements per bit")
    ap.add_argument("--meas-sweep", type=str, default=None,
                    help="comma list of measurement counts, e.g. 1,2,4,8; "
                         "runs the model suite at each and reports the trend")
    ap.add_argument("--no-perkey", action="store_true",
                    help="disable per-key feature normalization (ablation)")
    args = ap.parse_args()

    records = cds.load_dataset(args.dataset_dir)
    if len(records) == 0:
        print("No records in %s" % args.dataset_dir)
        return

    def _prep(max_meas=None):
        d = build_perbit_table(records, max_meas=max_meas)
        return d if args.no_perkey else perkey_normalize(d)

    # ---- measurement-count parameter sweep ---- #
    if args.meas_sweep:
        counts = [int(c) for c in args.meas_sweep.split(",") if c.strip()]
        trend = {}
        for mc in counts:
            data = _prep(max_meas=mc)
            n_keys = len(np.unique(data["groups"]))
            folds_k = min(args.folds, max(2, n_keys))
            folds = grouped_folds(data["groups"], folds_k, seed=args.seed)
            cfgs = {k: {**v, "epochs": args.epochs}
                    for k, v in default_model_suite().items()}
            res = run_models(data, folds, cfgs, seed=args.seed)
            trend[mc] = res
            _print_table(res, "meas_per_bit=%d (grouped %d-fold CV)"
                         % (mc, folds_k))
        out_path = os.path.join(args.dataset_dir, "eval_meas_sweep.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"dataset_dir": args.dataset_dir, "params": records.params,
                       "counts": counts, "perkey_norm": not args.no_perkey,
                       "trend": {str(mc): {k: {kk: {"mean": vv[0], "std": vv[1]}
                                                for kk, vv in v.items()}
                                            for k, v in res.items()}
                                 for mc, res in trend.items()}},
                      f, indent=2)
        print("\n[report] %s" % out_path)
        return

    data = _prep(max_meas=args.max_meas)
    n = len(data["y"])
    n_keys = len(np.unique(data["groups"]))
    pos = float(np.mean(data["y"]))
    print("[data] %d per-bit rows | %d keys | %d features | pos-rate %.3f | "
          "privileged=%s | torch=%s"
          % (n, n_keys, data["X"].shape[1], pos, data["has_priv"], _HAS_TORCH))
    if n_keys < args.folds:
        args.folds = max(2, n_keys)
        print("[data] few keys; reducing folds to %d" % args.folds)

    folds = grouped_folds(data["groups"], args.folds, seed=args.seed)

    if args.sweep:
        cfgs = {k: {**v, "epochs": args.epochs} for k, v in sweep_suite().items()}
        title = "HYPERPARAMETER SWEEP (grouped %d-fold CV)" % args.folds
        out_name = "eval_sweep.json"
    else:
        cfgs = {k: {**v, "epochs": args.epochs}
                for k, v in default_model_suite().items()}
        title = "MODEL COMPARISON (grouped %d-fold CV)" % args.folds
        out_name = "eval_report.json"

    results = run_models(data, folds, cfgs, seed=args.seed)
    _print_table(results, title)

    report = {
        "dataset_dir": args.dataset_dir,
        "params": records.params,
        "n_rows": n, "n_keys": n_keys, "pos_rate": pos,
        "folds": args.folds, "epochs": args.epochs,
        "has_privileged": bool(data["has_priv"]),
        "results": {k: {kk: {"mean": vv[0], "std": vv[1]}
                        for kk, vv in v.items()}
                    for k, v in results.items()},
    }
    out_path = os.path.join(args.dataset_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n[report] %s" % out_path)


if __name__ == "__main__":
    main()
