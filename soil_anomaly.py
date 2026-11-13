#!/usr/bin/env python3
"""
Trustworthy soil sensing: lightweight tamper/fault detection for root-zone IoT.

Real data : Metompkin Farm (train + in-site test) and Crisfield Farm (held-out site),
            METER TEROS 12 sensors, 15-min interval, SC / ST / WC per sensor unit.
Injected  : faults (spike, stuck, drift) and attacks (spoof, replay, poison) with
            known labels, because the public files contain no labelled attacks.
Detectors : rolling z-score, cross-sensor consistency rules, Isolation Forest,
            small autoencoder, and a hybrid (rules + Isolation Forest).

Usage:
  python soil_anomaly.py --met MetompkinFarm_*.csv --cris CrisfieldFarm_*.csv --out results
"""
import argparse, json, time, warnings, zlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

CH = ["SC", "ST", "WC"]            # conductivity, temperature, water content
DAY = 96                           # 15-min steps per day
ROLL = 7 * DAY                     # trailing window for local baseline (7 days)
TRAIN_END, VAL_END = 0.30, 0.40    # 0-30% fit, 30-40% clean calibration, 40-100% test
FAULTS = ["spike", "stuck", "drift"]
ATTACKS = ["spoof", "replay", "poison"]
ETYPES = FAULTS + ATTACKS
BASE_DETS = ["zscore", "rules", "iforest", "autoenc"]
DETS = BASE_DETS + ["hybrid"]
GUARD = DAY                        # rows after an event ignored when counting false alarms


# ----------------------------------------------------------------------------- data
def load_site(path, exclude=()):
    raw = pd.read_csv(path, dtype=str)
    raw.index = pd.to_datetime(raw.iloc[:, 0])
    raw = raw.drop(columns=raw.columns[0]).replace("NAN", np.nan)
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.mask(raw == 999)     # documented malfunction flag (absent in these files)
    units = {}
    for col in raw.columns:
        u, ch = col.rsplit("_", 1)
        units.setdefault(u, {})[ch] = raw[col]
    return {u: pd.DataFrame(d)[CH] for u, d in units.items() if u not in exclude}


def partner_of(u, units):
    loc, depth = u.rsplit("-", 1)
    other = loc + ("-S" if depth == "D" else "-D")
    return other if other in units else None


# ------------------------------------------------------------------------- features
def raw_features(df, partner):
    F = {}
    for c in CH:
        s = df[c]
        F[f"dev_{c}"] = s - s.rolling(ROLL, min_periods=DAY).median()
        F[f"d1_{c}"] = s.diff()
        F[f"lstd_{c}"] = np.log(s.rolling(8, min_periods=8).std() + 1e-3)
        d = s.diff()
        F[f"flat_{c}"] = (d == 0).astype(float).where(d.notna()).rolling(8, min_periods=8).mean()
    for c in CH:
        if partner is not None:
            x = df[c] - partner[c]
            F[f"xd_{c}"] = x - x.rolling(ROLL, min_periods=DAY).median()
        else:
            F[f"xd_{c}"] = pd.Series(0.0, index=df.index)
    return pd.DataFrame(F, index=df.index)


def add_physics_residual(F, ols=None, fit_rows=None):
    """Residual of each channel's local deviation predicted from the other two."""
    F = F.copy()
    if ols is None:
        ols = {}
        for c in CH:
            others = [f"dev_{o}" for o in CH if o != c]
            m = LinearRegression().fit(F.loc[fit_rows, others], F.loc[fit_rows, f"dev_{c}"])
            ols[c] = m
    for c in CH:
        others = [f"dev_{o}" for o in CH if o != c]
        ok = F[others + [f"dev_{c}"]].notna().all(axis=1)
        pr = pd.Series(np.nan, index=F.index)
        pr[ok] = F.loc[ok, f"dev_{c}"] - ols[c].predict(F.loc[ok, others])
        F[f"pr_{c}"] = pr
    return F, ols


class UnitRef:
    """Per-unit calibration fitted on the clean training slice only."""
    def __init__(self, df, partner):
        n = len(df)
        self.tr = slice(0, int(TRAIN_END * n))
        self.va = slice(int(TRAIN_END * n), int(VAL_END * n))
        self.te0 = int(VAL_END * n)
        F = raw_features(df, partner)
        tr_rows = F.index[self.tr]
        okrows = tr_rows[F.loc[tr_rows, [f"dev_{c}" for c in CH]].notna().all(axis=1)]
        F, self.ols = add_physics_residual(F, fit_rows=okrows)
        T = F.loc[tr_rows].dropna()
        self.med = T.median()
        rng90 = (T.quantile(0.95) - T.quantile(0.05)) / 3.29
        floor = pd.Series({c: (0.05 if c.startswith("flat_") else 1e-9) for c in T.columns})
        self.scale = np.maximum(rng90, floor).replace(0, np.nan).fillna(1.0)
        self.sd = {c: float(T[f"dev_{c}"].std()) for c in CH}     # attack magnitude unit
        self.cols = list(F.columns)

    def z(self, df, partner):
        F, _ = add_physics_residual(raw_features(df, partner), ols=self.ols)
        Zr = (F[self.cols] - self.med) / self.scale
        return np.sign(Zr) * np.log1p(Zr.abs())          # compress natural heavy tails


# ------------------------------------------------------------------------- detectors
def fit_models(Ztrain, seed):
    rng = np.random.default_rng(seed)
    Z = Ztrain.dropna()
    if len(Z) > 30000:
        Z = Z.iloc[rng.choice(len(Z), 30000, replace=False)]
    X = Z.values
    iso = IsolationForest(n_estimators=100, max_samples=256, random_state=seed).fit(X)
    ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), activation="tanh", max_iter=80,
                      random_state=seed).fit(X, X)
    return {"iso": iso, "ae": ae, "cols": list(Z.columns)}


def score(Z, M):
    out = pd.DataFrame(np.nan, index=Z.index, columns=BASE_DETS)
    ok = Z.notna().all(axis=1)
    A = Z[ok]
    X = A.values
    out.loc[ok, "zscore"] = A[[f"dev_{c}" for c in CH]].abs().max(axis=1)
    out.loc[ok, "rules"] = A[[f"xd_{c}" for c in CH] + [f"pr_{c}" for c in CH]
                                  + ["flat_SC", "flat_WC"]].abs().max(axis=1)
    out.loc[ok, "iforest"] = -M["iso"].score_samples(X)
    out.loc[ok, "autoenc"] = ((M["ae"].predict(X) - X) ** 2).mean(axis=1)
    return out


def thresholds(S_val, q):
    thr = {d: float(np.nanquantile(S_val[d], q)) for d in BASE_DETS}
    S = S_val.copy()
    S["hybrid"] = np.maximum(S["rules"] / thr["rules"], S["iforest"] / thr["iforest"])
    thr["hybrid"] = float(np.nanquantile(S["hybrid"], q))
    return thr


def add_hybrid(S, thr):
    S = S.copy()
    S["hybrid"] = np.maximum(S["rules"] / thr["rules"], S["iforest"] / thr["iforest"])
    return S


def alerts(S, thr, k):
    A = pd.DataFrame(False, index=S.index, columns=DETS)
    for d in DETS:
        above = (S[d] > thr[d]).astype(float).where(S[d].notna())
        A[d] = above.rolling(k, min_periods=k).min().fillna(0).astype(bool)
    return A


# ------------------------------------------------------------------------ injection
def inject(df, te0, ref, rng, sev, n_per_type):
    """Return corrupted copy, per-row type label (0 = clean) and event list."""
    X = df[CH].values.copy()
    n = len(X)
    valid = ~np.isnan(df[CH].values).any(axis=1)
    sd = np.array([ref.sd[c] for c in CH])
    lab = np.zeros(n, dtype=np.int8)
    events, busy = [], np.zeros(n, dtype=bool)
    order = [t for t in ETYPES for _ in range(n_per_type)]
    rng.shuffle(order)
    dur_rng = {"spike": (2, 6), "stuck": (8, 96), "drift": (DAY, 5 * DAY),
               "spoof": (DAY // 2, 3 * DAY), "replay": (DAY // 2, 3 * DAY),
               "poison": (3 * DAY, 10 * DAY)}
    for t in order:
        for _ in range(300):
            d = int(rng.integers(*dur_rng[t]))
            s = int(rng.integers(te0 + 2 * DAY, n - d - GUARD - 1))
            lo, hi = s - GUARD, s + d + 2 * GUARD
            lag = int(rng.integers(1, 11)) * DAY
            need_lo = s - lag if t == "replay" else s - 1
            if need_lo < ROLL or busy[lo:hi].any() or not valid[need_lo:s + d].all():
                continue
            break
        else:
            continue
        sl = slice(s, s + d)
        sign = rng.choice([-1.0, 1.0])
        if t == "spike":
            c = rng.integers(0, 3)
            X[sl, c] += sign * rng.uniform(3, 8) * sev * sd[c]
        elif t == "stuck":
            chs = rng.choice(3, size=int(rng.integers(1, 4)), replace=False)
            for c in chs:
                X[sl, c] = X[s - 1, c]
        elif t == "drift":
            c = rng.integers(0, 3)
            X[sl, c] += sign * np.linspace(0, rng.uniform(1.5, 4) * sev * sd[c], d)
        elif t == "spoof":
            chs = rng.choice(3, size=int(rng.integers(1, 3)), replace=False)
            for c in chs:
                X[sl, c] += sign * rng.uniform(1.5, 4) * sev * sd[c]
        elif t == "replay":
            X[sl] = df[CH].values[s - lag:s - lag + d]
        elif t == "poison":
            amp = rng.uniform(1, 3) * sev
            ramp = np.linspace(0, 1, d)
            X[sl, 2] += sign * ramp * amp * sd[2]       # WC
            X[sl, 0] += sign * ramp * amp * sd[0]       # SC moves with it
        lab[sl] = ETYPES.index(t) + 1
        busy[s:s + d] = True
        events.append((t, s, s + d))
    X[:, 2] = np.clip(X[:, 2], 0.01, 0.75)
    X[:, 0] = np.clip(X[:, 0], 0.0, None)
    X[:, 1] = np.clip(X[:, 1], -5, 50)
    Y = pd.DataFrame(X, index=df.index, columns=CH)
    return Y, lab, events


# ----------------------------------------------------------------------- evaluation
def evaluate_unit(u, units, refs, M, thr_cache, seed, sev, n_per_type, q, k):
    df = units[u]
    p = partner_of(u, units)
    pdf = units[p] if p else None
    ref = refs[u]
    rng = np.random.default_rng(seed)
    X, lab, events = inject(df, ref.te0, ref, rng, sev, n_per_type)
    S = add_hybrid(score(ref.z(X, pdf), M), thr_cache[u])
    A = alerts(S, thr_cache[u], k)
    n = len(df)
    te = np.arange(ref.te0, n)
    scored = S["rules"].notna().values
    guard = np.zeros(n, dtype=bool)
    for _, s, e in events:
        guard[e:e + GUARD] = True
    rows, ev_rows, pool = [], [], {}
    for d in DETS:
        a = A[d].values
        m_pos = (lab > 0) & scored
        m_neg = (lab == 0) & scored & ~guard
        m_pos[:ref.te0] = m_neg[:ref.te0] = False
        mm = m_pos | m_neg
        pool[d] = ((S[d].values[mm] / thr_cache[u][d]).astype(np.float32), m_pos[mm])
        tp, fn = int((a & m_pos).sum()), int((~a & m_pos).sum())
        fp, tn = int((a & m_neg).sum()), int((~a & m_neg).sum())
        # false-alert episodes on clean rows
        ac = a & m_neg
        eps = int(((ac.astype(int) - np.r_[0, ac[:-1].astype(int)]) == 1).sum())
        rows.append(dict(unit=u, det=d, tp=tp, fp=fp, fn=fn, tn=tn,
                         clean_days=m_neg.sum() / DAY, fa_episodes=eps))
        for t, s, e in events:
            hit = np.flatnonzero(a[s:e])
            ev_rows.append(dict(unit=u, det=d, type=t, kind="attack" if t in ATTACKS else "fault",
                                detected=bool(len(hit)), delay_steps=int(hit[0]) if len(hit) else np.nan,
                                dur=e - s))
    return rows, ev_rows, pool, (S, A, lab, events, X)


def summarize(rows, evrows, pools, sev, site, seeds):
    from sklearn.metrics import roc_auc_score, average_precision_score
    R = pd.DataFrame(rows)
    g = R.groupby("det")[["tp", "fp", "fn", "tn", "clean_days", "fa_episodes"]].sum()
    out = pd.DataFrame(index=g.index)
    out["precision"] = g.tp / (g.tp + g.fp).replace(0, np.nan)
    out["recall"] = g.tp / (g.tp + g.fn)
    out["f1"] = 2 * out.precision * out.recall / (out.precision + out.recall)
    out["fpr"] = g.fp / (g.fp + g.tn)
    out["false_alerts_per_30d_per_unit"] = g.fa_episodes / g.clean_days * 30
    E = pd.DataFrame(evrows)
    out["event_recall"] = E.groupby("det").detected.mean()
    out["median_delay_min"] = E[E.detected].groupby("det").delay_steps.median() * 15
    pa = {d: (np.concatenate([p[d][0] for p in pools]), np.concatenate([p[d][1] for p in pools]))
          for d in DETS}
    out["auroc"] = pd.Series({d: roc_auc_score(pa[d][1], pa[d][0]) for d in DETS})
    out["avg_precision"] = pd.Series({d: average_precision_score(pa[d][1], pa[d][0]) for d in DETS})
    out["site"], out["severity"] = site, sev
    bytype = (E.groupby(["det", "type"]).detected.mean().unstack()[ETYPES]
              .assign(site=site, severity=sev))
    return out.loc[DETS], bytype.loc[DETS]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--met", required=True)
    ap.add_argument("--cris", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--severities", type=float, nargs="+", default=[1.0])
    ap.add_argument("--n-per-type", type=int, default=6)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--exclude-met", nargs="*", default=["VFMS-SP-S"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

    sites = {"Metompkin": load_site(a.met, a.exclude_met), "Crisfield": load_site(a.cris)}
    refs, thr_cache, Ztrain = {}, {}, []
    for site, units in sites.items():
        refs[site] = {u: UnitRef(units[u], units[partner_of(u, units)] if partner_of(u, units) else None)
                      for u in units}
    for u, df in sites["Metompkin"].items():           # train pooled models on Metompkin only
        p = partner_of(u, sites["Metompkin"])
        Z = refs["Metompkin"][u].z(df, sites["Metompkin"][p] if p else None)
        Ztrain.append(Z.iloc[refs["Metompkin"][u].tr])
    t0 = time.time()
    M = fit_models(pd.concat(Ztrain), seed=0)
    print(f"models fitted in {time.time()-t0:.1f}s on {sum(len(z.dropna()) for z in Ztrain)} clean rows")

    for site, units in sites.items():                   # clean-validation thresholds per unit
        thr_cache[site] = {}
        for u, df in units.items():
            p = partner_of(u, units)
            r = refs[site][u]
            S = score(r.z(df, units[p] if p else None).iloc[r.va], M)
            thr_cache[site][u] = thresholds(S, a.quantile)

    summ, typ, ex = [], [], None
    for sev in a.severities:
        for site, units in sites.items():
            rows, evrows, pools = [], [], []
            for sd in range(a.seeds):
                for u in units:
                    r, e, pl, extra = evaluate_unit(u, units, refs[site], M, thr_cache[site],
                                                1000 * sd + zlib.crc32(u.encode()) % 997, sev, a.n_per_type,
                                                a.quantile, a.persist)
                    rows += r; evrows += e; pools.append(pl)
                    if ex is None and site == "Metompkin" and sev == 1.0 and len(extra[3]) >= 6:
                        ex = (u, extra)
            s, t = summarize(rows, evrows, pools, sev, site, a.seeds)
            summ.append(s.reset_index()); typ.append(t.reset_index())
            print(f"\n[{site}] severity {sev}\n", s.round(3).drop(columns=["site", "severity"]).to_string())
            pd.DataFrame(evrows).to_csv(out / f"events_{site}_sev{sev}.csv", index=False)
    pd.concat(summ).to_csv(out / "summary.csv", index=False)
    pd.concat(typ).to_csv(out / "recall_by_type.csv", index=False)

    # edge cost: scoring time and model size
    site0 = sites["Metompkin"]; u0 = next(iter(site0)); p0 = partner_of(u0, site0)
    Z0 = refs["Metompkin"][u0].z(site0[u0], site0[p0] if p0 else None).dropna().iloc[:10000]
    cost = {}
    for name, fn in [("zscore", lambda Z: Z[[f"dev_{c}" for c in CH]].abs().max(axis=1)),
                     ("rules", lambda Z: Z[[f"xd_{c}" for c in CH] + [f"pr_{c}" for c in CH]].abs().max(axis=1)),
                     ("iforest", lambda Z: -M["iso"].score_samples(Z.values.clip(-10, 10))),
                     ("autoenc", lambda Z: M["ae"].predict(Z.values.clip(-10, 10)))]:
        t0 = time.time(); fn(Z0); cost[name] = (time.time() - t0) / len(Z0) * 1e6
    cost = {"scoring_us_per_row": cost,
            "iforest_trees": 100, "iforest_max_samples": 256,
            "autoenc_params": int(sum(w.size for w in M["ae"].coefs_) + sum(b.size for b in M["ae"].intercepts_)),
            "n_features": len(M["cols"])}
    json.dump(cost, open(out / "edge_cost.json", "w"), indent=2)
    json.dump(vars(a) | {"split": [TRAIN_END, VAL_END]}, open(out / "config.json", "w"), indent=2)
    print("\nedge cost:", json.dumps(cost, indent=1))

    # figures
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = pd.concat(typ); T = T[T.severity == a.severities[0]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, site in zip(axes, ["Metompkin", "Crisfield"]):
        sub = T[T.site == site].set_index("det")[ETYPES]
        sub.plot.bar(ax=ax, width=0.8, legend=False)
        ax.set_title(f"{site} ({'in-site' if site == 'Metompkin' else 'held-out site'})")
        ax.set_xlabel(""); ax.set_ylabel("event recall"); ax.tick_params(axis="x", rotation=30)
    axes[1].legend(ETYPES, fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(out / "fig_recall_by_type.pdf"); plt.savefig(out / "fig_recall_by_type.png", dpi=200)
    plt.close()
    if ex:
        u, (S, A, lab, events, X) = ex
        t, s, e = next(ev for ev in events if ev[0] == "spoof") if any(ev[0] == "spoof" for ev in events) else events[0]
        lo, hi = max(s - 3 * DAY, 0), min(e + 3 * DAY, len(X))
        fig, ax = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True)
        ax[0].plot(X.index[lo:hi], sites["Metompkin"][u]["WC"].iloc[lo:hi], lw=0.8, label="original")
        ax[0].plot(X.index[lo:hi], X["WC"].iloc[lo:hi], lw=0.8, label="corrupted")
        ax[0].set_ylabel("WC (m$^3$/m$^3$)"); ax[0].legend(fontsize=7)
        ax[1].plot(S.index[lo:hi], S["rules"].iloc[lo:hi], lw=0.8, label="rules")
        ax[1].axhline(thr_cache["Metompkin"][u]["rules"], color="r", ls="--", lw=0.8)
        ax[1].set_ylabel("rule score")
        ax[2].plot(S.index[lo:hi], S["zscore"].iloc[lo:hi], lw=0.8, color="C2")
        ax[2].axhline(thr_cache["Metompkin"][u]["zscore"], color="r", ls="--", lw=0.8)
        ax[2].set_ylabel("z-score")
        for x in ax: x.axvspan(X.index[s], X.index[min(e, len(X) - 1)], color="orange", alpha=0.2)
        ax[0].set_title(f"Example: {t} event on {u}")
        plt.tight_layout(); plt.savefig(out / "fig_example.pdf"); plt.savefig(out / "fig_example.png", dpi=200)


if __name__ == "__main__":
    main()
