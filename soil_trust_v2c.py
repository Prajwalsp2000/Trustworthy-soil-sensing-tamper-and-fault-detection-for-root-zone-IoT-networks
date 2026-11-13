#!/usr/bin/env python3
"""
Trustworthy soil sensing, version 2b (matched-alarm protocol)
=============================================================
Same detectors as v2, plus: (1) every detector is calibrated to the SAME target false-alarm
rate on the clean validation slice, so recall comparisons are fair and deployable;
(2) --roll-days sets the local baseline window; (3) --only-site / --no-figs for tuning runs.

Original v2 description:

Lightweight, physics-aware, multi-sensor tamper/fault detection for root-zone IoT.

Needs soil_anomaly.py (version 1) in the same folder: it supplies the data loader,
the fault/attack injector and the four baseline detectors, so both versions are
evaluated on exactly the same injected events.

v2 has four tiers (each is cheap enough for an edge gateway):
  T1_peer    stateless  : residual of a sensor predicted from every other sensor on the farm
                          (shared natural events such as rain cancel out) plus
                          cross-channel physics terms
  T2_window  stateful   : windowed mean-shift evidence (12 h / 2 d / 6 d) on the same residuals
                          (catches slow drift and low-and-slow poisoning)
  T3_replay  exact-repeat check: a rolling window that repeats an earlier window of
                          the same sensor to the last digit is a replay
  T4_flat    per-sensor flat-run model: run length of an unchanged value compared with
                          that sensor's own normal run lengths (stuck-at)
The tiers are fused with a logical OR; each tier gets a share of the false-alarm budget.
Ablations remove one component at a time.

Run:
  python soil_trust_v2.py --met MetompkinFarm_WC_ST_SC_220719-240723.csv \
                          --cris CrisfieldFarm_WC_ST_SC_210714-240711.csv \
                          --out results_v2 --seeds 3 --severities 0.5 1.0 2.0
Quick test (a few minutes):  add  --seeds 1 --severities 1.0
"""
import argparse, json, time, zlib, warnings, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import soil_anomaly as sa

warnings.filterwarnings("ignore")
CH, DAY, GUARD = sa.CH, sa.DAY, sa.GUARD
SLOW_ROLL = 60 * DAY                    # long baseline for the slow depth-gap tier
SLOW_WINDOWS = (2 * DAY, 6 * DAY)
ROLL2 = 14 * DAY                        # trailing baseline window (set by --roll-days)
WINDOWS = (DAY // 2, 2 * DAY, 6 * DAY)  # evidence windows for the stateful tier (12 h, 2 d, 6 d)
CUSUM_CLIP = 6.0                        # winsorise residuals (in residual sigmas) before accumulating
FLAT_Q = 0.9995                         # normal flat-run quantile learned on training data
V1 = ["zscore", "rules", "iforest", "autoenc", "hybrid"]
TIERS = ["T1_peer", "T2_window", "T3_replay", "T4_flat", "T5_slow", "T1n_nopeer", "T2n_nopeer"]
SINGLE = ["T1_peer", "T2_window", "T3_replay", "T4_flat", "T5_slow"]
VARIANTS = {
    "v2_full":        ["T1_peer", "T2_window", "T3_replay", "T4_flat"],
    "v2_with_slow":   ["T1_peer", "T2_window", "T3_replay", "T4_flat", "T5_slow"],
    "v2_no_peer":     ["T1n_nopeer", "T2n_nopeer", "T3_replay", "T4_flat"],
    "v2_no_stateful": ["T1_peer", "T3_replay", "T4_flat"],
    "v2_no_replay":   ["T1_peer", "T2_window", "T4_flat"],
    "v2_no_flat":     ["T1_peer", "T2_window", "T3_replay"],
}
PERSIST = {"T1_peer": 2, "T2_window": 1, "T3_replay": 2, "T4_flat": 1, "T5_slow": 1,
           "T1n_nopeer": 2, "T2n_nopeer": 1}
COMBO = "v2_plus_ml"                     # v2 tiers OR the v1 hybrid (learned) detector
# specialised checks (replay, flat-run, ...) OR generic ML detectors: name -> (tiers, ml detectors)
EXTRA = {
    "t34_autoenc":     (["T3_replay", "T4_flat"], ["autoenc"]),
    "t34_iforest":     (["T3_replay", "T4_flat"], ["iforest"]),
    "t34_ae_if":       (["T3_replay", "T4_flat"], ["autoenc", "iforest"]),
    "v2_plus_autoenc": (["T1_peer", "T2_window", "T3_replay", "T4_flat"], ["autoenc"]),
}
ALL_DETS = V1 + SINGLE + list(VARIANTS) + [COMBO] + list(EXTRA)
MAIN = ["zscore", "rules", "iforest", "autoenc", "hybrid", "t34_autoenc", "v2_full", "v2_plus_autoenc"]
OPC_DETS = list(dict.fromkeys(MAIN + ["v2_no_peer", "v2_no_stateful", "v2_with_slow"] + list(EXTRA)))
MULTS = [float(x) for x in np.geomspace(0.3, 8.0, 28)]
MGRID = np.geomspace(0.3, 8.0, 36)                        # multipliers searched for FPR matching
TARGETS = [0.005, 0.01, 0.02, 0.04]                         # target clean-validation false-alarm rates   # threshold multipliers for operating curves
LABEL = {"zscore": "Rolling z-score", "rules": "Cross-sensor rules", "iforest": "Isolation Forest",
         "autoenc": "Autoencoder", "hybrid": "Rules + Isolation Forest",
         "v2_full": "v2 (peer, windowed, replay, flat-run)", COMBO: "v2 + rules + Isolation Forest",
         "T1_peer": "Peer residual only", "T2_window": "Windowed evidence only",
         "T3_replay": "Replay check only", "T4_flat": "Flat-run check only",
         "T5_slow": "Slow depth-gap only", "v2_with_slow": "v2 + slow depth-gap tier",
         "v2_no_peer": "v2 without peer regression", "v2_no_stateful": "v2 without windowed evidence",
         "v2_no_replay": "v2 without replay check", "v2_no_flat": "v2 without flat-run check",
         "t34_autoenc": "Replay + flat-run + autoencoder", "t34_iforest": "Replay + flat-run + Isolation Forest",
         "t34_ae_if": "Replay + flat-run + AE + IF", "v2_plus_autoenc": "v2 + autoencoder"}
COLORS = {"t34_autoenc": "#d6616b", "t34_iforest": "#e7ba52", "t34_ae_if": "#8c6d31", "v2_plus_autoenc": "#843c39",
          "zscore": "#9e9e9e", "rules": "#6baed6", "iforest": "#3182bd", "autoenc": "#08519c",
          "hybrid": "#74c476", "v2_full": "#e6550d", COMBO: "#756bb1"}


# ------------------------------------------------------------------ site context
class SiteCtx:
    """Clean per-unit local deviations and farm-consensus signals for one site."""
    def __init__(self, units):
        self.units = units
        self.dev = {u: df[CH] - df[CH].rolling(ROLL2, min_periods=DAY).median()
                    for u, df in units.items()}

    def loc_gaps(self):
        """Clean shallow-minus-deep gap per location, per channel."""
        out = {}
        for u, df in self.units.items():
            loc, depth = u.rsplit("-", 1)
            p = sa.partner_of(u, self.units)
            if depth == "S" and p:
                out[loc] = df[CH] - self.units[p][CH]
        return out

    def other_gap(self, u):
        """Mean clean S-D gap over the other locations of the farm."""
        loc = u.rsplit("-", 1)[0]
        gs = [g for l, g in self.loc_gaps().items() if l != loc]
        return sum(g.fillna(0.0) for g in gs) / max(len(gs), 1) if gs else None

    def peers(self, u):
        """For each channel: DataFrame of the other units' clean local deviations (NaN -> 0)."""
        return {c: pd.concat({p: d[c] for p, d in self.dev.items() if p != u}, axis=1).fillna(0.0)
                for c in CH}


def cusum(d):
    C = np.cumsum(d, axis=0)
    return C - np.minimum(0.0, np.minimum.accumulate(C, axis=0))


def dup_flag(X, W=8, min_changes=4):
    """True where the last W rows repeat an earlier (>= 1 day older) window exactly."""
    V = np.round(X.values * 1000.0)
    n = len(V)
    valid_row = ~np.isnan(V).any(axis=1)
    Vi = np.nan_to_num(V).astype(np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = (Vi[:, 0] * np.uint64(0x9E3779B97F4A7C15) + Vi[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
             + Vi[:, 2] * np.uint64(0x165667B19E3779F9))
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
        Hw = np.zeros(n, dtype=np.uint64)
        for i in range(W):
            Hw += np.roll(h, i) * np.uint64(2 * (i + 1) * 0x9E3779B1 + 1)
    chg = np.r_[0, (np.diff(V, axis=0) != 0).any(axis=1).astype(int)]
    cs = np.cumsum(chg)
    cb = np.cumsum(~valid_row)
    ok = np.zeros(n, bool)
    ok[W:] = ((cs[W:] - cs[:-W]) >= min_changes) & ((cb[W:] - cb[:-W]) == 0)
    Hl, okl = Hw.tolist(), ok.tolist()
    dup, seen = np.zeros(n, bool), set()
    for t in range(W, n):
        j = t - DAY
        if j >= W and okl[j]:
            seen.add(Hl[j])
        if okl[t] and Hl[t] in seen:
            dup[t] = True
    return dup, valid_row


class UnitV2:
    """Per-unit v2 calibration fitted on the clean training slice only."""
    def __init__(self, u, ctx, ref):
        self.u, self.ctx, self.ref = u, ctx, ref
        df = ctx.units[u]
        p = sa.partner_of(u, ctx.units)
        self.peerdev = ctx.peers(u)
        self.partner_df = ctx.units[p] if p else None
        self.sign = 1.0 if u.endswith("-S") else -1.0
        self.other_gap = ctx.other_gap(u)
        self.pdev = ctx.dev[p] if p else None
        D = ctx.dev[u]
        tr = ref.tr
        self.models, self.scale = {}, {}
        for variant in ("peer", "nopeer"):
            for c in CH:
                X, y = self.design(D, c, variant), D[c]
                ok = (X.notna().all(axis=1) & y.notna()).iloc[tr]
                Xt, yt = X.iloc[tr][ok], y.iloc[tr][ok]
                m = LinearRegression().fit(Xt, yt)
                res = yt - m.predict(Xt)
                sc = (res.quantile(0.95) - res.quantile(0.05)) / 3.29
                self.models[(variant, c)] = m
                self.scale[(variant, c)] = sc if sc > 1e-9 else 1.0
        self.slow_scale = {}
        if self.partner_df is not None:
            gap = self.gap_dev(df)
            for c in CH:
                g = gap[c].iloc[tr].dropna()
                sc = (g.quantile(0.95) - g.quantile(0.05)) / 3.29
                self.slow_scale[c] = sc if sc > 1e-9 else 1.0
        self.L = {}
        for c in CH:
            run = self.runs(df[c]).iloc[tr]
            self.L[c] = max(3.0, float(run.quantile(FLAT_Q)))

    def design(self, D, c, variant):
        cols = {f"own_{o}": D[o] for o in CH if o != c}
        cols["partner"] = self.pdev[c] if self.pdev is not None else 0.0
        if variant == "peer":
            for p, col in self.peerdev[c].items():
                cols[f"peer_{p}"] = col
        return pd.DataFrame(cols, index=D.index)

    def gap_dev(self, Xdf):
        """Depth-gap series (sensor minus its clean depth partner) minus its own 60-day trailing median."""
        g = self.sign * (Xdf[CH] - self.partner_df[CH])      # always shallow minus deep
        if self.other_gap is not None:
            g = g - self.other_gap                          # shared seasonal behaviour cancels
        return g - g.rolling(SLOW_ROLL, min_periods=DAY).median()

    @staticmethod
    def runs(s):
        same = (s.diff() == 0)
        grp = (~same).cumsum()
        return same.astype(int).groupby(grp).cumsum()

    def tiers(self, Xdf):
        """Raw tier scores for a (clean or injected) series of this unit."""
        D = Xdf[CH] - Xdf[CH].rolling(ROLL2, min_periods=DAY).median()
        out = pd.DataFrame(index=Xdf.index)
        for variant, tag in (("peer", ""), ("nopeer", "n")):
            Z = pd.DataFrame(index=Xdf.index)
            for c in CH:
                X, y = self.design(D, c, variant), D[c]
                ok = X.notna().all(axis=1) & y.notna()
                r = pd.Series(np.nan, index=Xdf.index)
                r[ok] = y[ok] - self.models[(variant, c)].predict(X[ok])
                Z[c] = r / self.scale[(variant, c)]
            s1 = Z.abs().max(axis=1)
            zc = Z.clip(-CUSUM_CLIP, CUSUM_CLIP).fillna(0.0)
            s2 = pd.concat([zc.rolling(w, min_periods=w).mean().abs() for w in WINDOWS], axis=1).max(axis=1)
            s2 = s2.where(s1.notna())
            if variant == "peer":
                out["T1_peer"], out["T2_window"] = s1, s2
            else:
                out["T1n_nopeer"], out["T2n_nopeer"] = s1, s2
        if self.partner_df is not None:
            gd = self.gap_dev(Xdf)
            zs = pd.concat([(gd[c] / self.slow_scale[c]).clip(-CUSUM_CLIP, CUSUM_CLIP) for c in CH], axis=1).fillna(0.0)
            s5 = pd.concat([zs.rolling(w, min_periods=w).mean().abs() for w in SLOW_WINDOWS], axis=1).max(axis=1)
            out["T5_slow"] = s5.where(gd.notna().all(axis=1))
        else:
            out["T5_slow"] = np.nan
        dup, valid_row = dup_flag(Xdf[CH])
        out["T3_replay"] = pd.Series(dup.astype(float), index=Xdf.index).where(valid_row)
        flat = pd.concat([self.runs(Xdf[c]) / self.L[c] for c in CH], axis=1).max(axis=1)
        out["T4_flat"] = flat.where(valid_row)
        return out[TIERS]


# ----------------------------------------------------------------- alert helpers
def alert_from_norm(s, k):
    above = (s > 1.0).astype(float).where(s.notna())
    return above.rolling(k, min_periods=k).min().fillna(0).astype(bool).values


# ------------------------------------------------------------------- evaluation
def compute_norm(X, pdf, ref, ur, M, thr1, thr2):
    """Normalised (score / threshold) series for every detector, for series X of one unit."""
    S1 = sa.add_hybrid(sa.score(ref.z(X, pdf), M), thr1)
    norm = {d: S1[d] / thr1[d] for d in V1}
    T = ur.tiers(X)
    for t in TIERS:
        norm[t] = T[t] / thr2[t]
    ok = norm["T1_peer"].notna()
    for v, ts in VARIANTS.items():
        norm[v] = pd.concat([norm[t] for t in ts], axis=1).max(axis=1, skipna=True).where(ok)
    norm[COMBO] = pd.concat([norm["v2_full"], norm["hybrid"]], axis=1).max(axis=1, skipna=True).where(ok)
    for name, (ts, ml) in EXTRA.items():
        norm[name] = pd.concat([norm[x] for x in ts + ml], axis=1).max(axis=1, skipna=True).where(ok)
    return norm, S1


def alert_at(norm, d, m, k_v1):
    """Alert array of detector d when its threshold is scaled by m."""
    if d in EXTRA:
        ts, ml = EXTRA[d]
        a = np.any([alert_from_norm(norm[t] / m, PERSIST[t]) for t in ts], axis=0)
        for x in ml:
            a = a | alert_from_norm(norm[x] / m, k_v1)
        return a
    if d in VARIANTS or d == COMBO:
        ts = VARIANTS["v2_full" if d == COMBO else d]
        a = np.any([alert_from_norm(norm[t] / m, PERSIST[t]) for t in ts], axis=0)
        if d == COMBO:
            a = a | alert_from_norm(norm["hybrid"] / m, k_v1)
        return a
    if d in TIERS:
        return alert_from_norm(norm[d] / m, PERSIST[d])
    return alert_from_norm(norm[d] / m, k_v1)


def calibrate_mstar(normc, ref, k_v1):
    """Per detector, the smallest threshold multiplier whose clean-validation FPR <= target."""
    valid = normc["T1_peer"].notna().values
    va = ref.va
    stop = va.stop
    sub = {d: v.iloc[:stop] for d, v in normc.items()}
    vv = valid[va]
    out = {}
    for d in OPC_DETS:
        f = np.array([(alert_at(sub, d, m, k_v1)[va] & vv).sum() / max(vv.sum(), 1) for m in MGRID])
        for tg in TARGETS:
            ok_idx = [i for i in range(len(MGRID)) if np.all(f[i:] <= tg)]
            out[(d, tg)] = float(MGRID[ok_idx[0]]) if ok_idx else float(MGRID[-1])
    return out


def evaluate_unit(u, site, ctx, ref, ur, M, thr1, thr2, mstar, seed, sev, n_per_type, k_v1, capture):
    units = ctx.units
    df = units[u]
    p = sa.partner_of(u, units)
    pdf = units[p] if p else None
    rng = np.random.default_rng(seed)
    X, lab, events = sa.inject(df, ref.te0, ref, rng, sev, n_per_type)
    norm, S1 = compute_norm(X, pdf, ref, ur, M, thr1, thr2)
    alert = {d: alert_at(norm, d, 1.0, k_v1) for d in ALL_DETS}

    n = len(df)
    scored = norm["T1_peer"].notna().values & S1["rules"].notna().values
    guard = np.zeros(n, bool)
    for _, s, e in events:
        guard[e:e + GUARD] = True
    delta = (X[CH].values - df[CH].values) / np.array([ref.sd[c] for c in CH])
    delta = np.nan_to_num(np.abs(delta)).sum(axis=1)
    m_pos = (lab > 0) & scored
    m_neg = (lab == 0) & scored & ~guard
    m_pos[:ref.te0] = m_neg[:ref.te0] = False
    mm = m_pos | m_neg

    def episodes(a):
        ac = a & m_neg
        return int(((ac.astype(int) - np.r_[0, ac[:-1].astype(int)]) == 1).sum())

    def event_rows(a, d):
        out = []
        for t, s, e in events:
            hit = np.flatnonzero(a[s:e])
            tot = delta[s:e].sum()
            first = int(hit[0]) if len(hit) else e - s
            out.append(dict(unit=u, det=d, type=t, detected=bool(len(hit)),
                            delay_steps=int(hit[0]) if len(hit) else np.nan,
                            exposure_frac=delta[s:s + first].sum() / tot if tot > 0 else np.nan, dur=e - s))
        return out

    rows, evrows, pool = [], [], {}
    for d in ALL_DETS:
        a = alert[d]
        pool[d] = (norm[d].values[mm].astype(np.float32), m_pos[mm])
        tp, fn = int((a & m_pos).sum()), int((~a & m_pos).sum())
        fp, tn = int((a & m_neg).sum()), int((~a & m_neg).sum())
        rows.append(dict(unit=u, det=d, tp=tp, fp=fp, fn=fn, tn=tn,
                         clean_days=m_neg.sum() / DAY, fa_episodes=episodes(a)))
        evrows += event_rows(a, d)

    curve = []
    for d in OPC_DETS:
        for m in MULTS:
            a = alert_at(norm, d, m, k_v1)
            curve.append(dict(unit=u, det=d, mult=m, eps=episodes(a), fp=int((a & m_neg).sum()),
                              neg=int(m_neg.sum()), clean_days=m_neg.sum() / DAY, n_ev=len(events),
                              n_det=int(sum(a[s_:e_].any() for _, s_, e_ in events))))
    mrows, mfa = [], []
    for d in OPC_DETS:
        for tg in TARGETS:
            a = alert_at(norm, d, mstar[(d, tg)], k_v1)
            mfa.append(dict(unit=u, det=d, target=tg, fp=int((a & m_neg).sum()), neg=int(m_neg.sum()),
                            eps=episodes(a), clean_days=m_neg.sum() / DAY))
            mrows += [dict(r, target=tg) for r in event_rows(a, d)]
    extra = (dict(df=df, X=X, events=events,
                  norm={d: norm[d] for d in ("zscore", "rules", "T1_peer", "T2_window", "v2_full")})
             if capture else None)
    return rows, evrows, pool, curve, mrows, mfa, extra


def summarize(rows, evrows, pools, sev, site):
    R = pd.DataFrame(rows)
    g = R.groupby("det")[["tp", "fp", "fn", "tn", "clean_days", "fa_episodes"]].sum().loc[ALL_DETS]
    out = pd.DataFrame(index=g.index)
    out["precision"] = g.tp / (g.tp + g.fp).replace(0, np.nan)
    out["recall_rows"] = g.tp / (g.tp + g.fn)
    out["f1_rows"] = 2 * out.precision * out.recall_rows / (out.precision + out.recall_rows)
    out["fpr"] = g.fp / (g.fp + g.tn)
    out["false_alerts_per_30d_per_unit"] = g.fa_episodes / g.clean_days * 30
    E = pd.DataFrame(evrows)
    out["event_recall"] = E.groupby("det").detected.mean()
    out["median_delay_min"] = E[E.detected].groupby("det").delay_steps.median() * 15
    out["mean_undetected_exposure"] = E.groupby("det").exposure_frac.mean()
    pa = {d: (np.nan_to_num(np.concatenate([p[d][0] for p in pools]), nan=0.0), np.concatenate([p[d][1] for p in pools]))
          for d in ALL_DETS}
    out["auroc"] = pd.Series({d: roc_auc_score(pa[d][1], pa[d][0]) for d in ALL_DETS})
    out["avg_precision"] = pd.Series({d: average_precision_score(pa[d][1], pa[d][0]) for d in ALL_DETS})
    out["site"], out["severity"] = site, sev
    bt = E.groupby(["det", "type"]).detected.mean().unstack()[sa.ETYPES].loc[ALL_DETS]
    bt = bt.assign(site=site, severity=sev)
    return out.reset_index().rename(columns={"index": "det"}), bt.reset_index()


def oracle_table(C):
    """Event recall at fixed realised test FPR, interpolated on each detector's operating curve."""
    rows = []
    for (site, sev, d), c in C.groupby(["site", "severity", "det"]):
        g = c.groupby("fpr").event_recall.max().sort_index()
        x = np.log10(np.clip(g.index.values, 1e-5, None))
        y = np.maximum.accumulate(g.values)
        for tg in TARGETS:
            rows.append(dict(site=site, severity=sev, det=d, target_fpr=tg,
                             event_recall=float(np.interp(np.log10(tg), x, y))))
    return pd.DataFrame(rows)


def summarize_matched(mrows, mfa, sev, site):
    E, F = pd.DataFrame(mrows), pd.DataFrame(mfa)
    g = F.groupby(["det", "target"])[["fp", "neg", "eps", "clean_days"]].sum()
    out = pd.DataFrame(index=g.index)
    out["realised_fpr"] = g.fp / g.neg
    out["false_alerts_per_30d_per_unit"] = g.eps / g.clean_days * 30
    out["event_recall"] = E.groupby(["det", "target"]).detected.mean()
    out["median_delay_min"] = E[E.detected].groupby(["det", "target"]).delay_steps.median() * 15
    out["mean_undetected_exposure"] = E.groupby(["det", "target"]).exposure_frac.mean()
    out["site"], out["severity"] = site, sev
    bt = E.groupby(["det", "target", "type"]).detected.mean().unstack()[sa.ETYPES].assign(site=site, severity=sev)
    return out.reset_index(), bt.reset_index()


# ------------------------------------------------------------------------ plots
def legend_below(fig, ax, ncol=3, fs=7):
    """Place the legend under the plots so it never covers data."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=fs, frameon=False)
    fig._legend_below = True


def finish_layout(fig):
    fig.tight_layout()
    if getattr(fig, "_legend_below", False):
        fig.subplots_adjust(bottom=0.30)


def make_figures(S, B, ex, out, C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    sites = ["Metompkin", "Crisfield"]
    sev0 = 1.0 if 1.0 in S.severity.unique() else S.severity.unique()[0]
    S1 = S[S.severity == sev0]

    def save(fig, name):
        finish_layout(fig)
        fig.savefig(out / f"{name}.png", dpi=220)
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)

    # 1 AUROC and average precision
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a, met, ttl in zip(ax, ["auroc", "avg_precision"], ["AUROC (row level)", "Average precision (row level)"]):
        w = 0.38
        for i, site in enumerate(sites):
            d = S1[(S1.site == site)].set_index("det").loc[MAIN, met]
            a.bar(np.arange(len(MAIN)) + (i - 0.5) * w, d.values, w, label=site if met == "auroc" else None,
                  color=[COLORS[m] for m in MAIN], alpha=1.0 if i == 0 else 0.55, edgecolor="k", linewidth=0.4)
        a.set_xticks(range(len(MAIN)))
        a.set_xticklabels([LABEL[m] for m in MAIN], rotation=30, ha="right")
        a.set_title(ttl)
    ax[0].text(0.01, 0.97, "solid = Metompkin (in-site), light = Crisfield (held-out)", transform=ax[0].transAxes,
               fontsize=7, va="top")
    save(fig, "fig1_auroc_ap")

    # 2 event recall heatmap by event type
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    for a, site in zip(ax, sites):
        H = B[(B.severity == sev0) & (B.site == site)].set_index("det").loc[MAIN, sa.ETYPES]
        im = a.imshow(H.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        a.set_xticks(range(len(sa.ETYPES)))
        a.set_xticklabels(sa.ETYPES)
        a.set_yticks(range(len(MAIN)))
        a.set_yticklabels([LABEL[m] for m in MAIN])
        a.set_title(f"{site}: event recall by type")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                a.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if H.values[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025)
    save(fig, "fig2_recall_by_type")

    # 3 severity curve
    if S.severity.nunique() > 1:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
        for a, site in zip(ax, sites):
            for m in MAIN:
                d = S[(S.site == site) & (S.det == m)].sort_values("severity")
                a.plot(d.severity, d.event_recall, marker="o", color=COLORS[m],
                       lw=2.4 if m == "v2_full" else 1.2, label=LABEL[m])
            a.set_xscale("log", base=2)
            a.set_xticks(sorted(S.severity.unique()))
            a.set_xticklabels([str(s) for s in sorted(S.severity.unique())])
            a.set_xlabel("attack/fault severity multiplier")
            a.set_title(site)
        ax[0].set_ylabel("event recall")
        ax[1].legend(fontsize=7)
        save(fig, "fig3_severity_curve")

    # 4 recall vs false alarms
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for a, site in zip(ax, sites):
        d = S1[S1.site == site].set_index("det")
        for m in MAIN:
            a.scatter(d.loc[m, "false_alerts_per_30d_per_unit"], d.loc[m, "event_recall"], s=70,
                      color=COLORS[m], edgecolor="k", label=LABEL[m], zorder=3)
        a.set_xlabel("false alert episodes per sensor per 30 days")
        a.set_title(site)
    ax[0].set_ylabel("event recall")
    ax[1].legend(fontsize=7)
    save(fig, "fig4_recall_vs_false_alerts")

    # 5 ablation
    ab = list(VARIANTS) + SINGLE
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for a, site in zip(ax, sites):
        d = S1[S1.site == site].set_index("det").loc[ab, "event_recall"]
        a.bar(range(len(ab)), d.values, color=["#e6550d"] + ["#fdae6b"] * (len(VARIANTS) - 1) + ["#bdbdbd"] * len(SINGLE), edgecolor="k", linewidth=0.4)
        a.set_xticks(range(len(ab)))
        a.set_xticklabels([LABEL[x] for x in ab], rotation=40, ha="right")
        a.set_title(f"{site}: ablation (event recall)")
    ax[0].set_ylabel("event recall")
    save(fig, "fig5_ablation")

    # 6 delay and undetected exposure
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    w = 0.38
    for i, site in enumerate(sites):
        d = S1[S1.site == site].set_index("det").loc[MAIN]
        ax[0].bar(np.arange(len(MAIN)) + (i - 0.5) * w, d.median_delay_min / 60.0, w,
                  color=[COLORS[m] for m in MAIN], alpha=1.0 if i == 0 else 0.55, edgecolor="k", linewidth=0.4)
        ax[1].bar(np.arange(len(MAIN)) + (i - 0.5) * w, d.mean_undetected_exposure, w,
                  color=[COLORS[m] for m in MAIN], alpha=1.0 if i == 0 else 0.55, edgecolor="k", linewidth=0.4)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("median detection delay (hours)")
    ax[1].set_ylabel("share of attack-induced bias undetected")
    for a in ax:
        a.set_xticks(range(len(MAIN)))
        a.set_xticklabels([LABEL[m] for m in MAIN], rotation=30, ha="right")
    ax[0].set_title("Detection delay (solid Metompkin, light Crisfield)")
    ax[1].set_title("Undetected exposure (lower is better)")
    save(fig, "fig6_delay_and_exposure")

    # 8 operating curves (event recall vs false alerts as the threshold is scaled)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for a, site in zip(ax, sites):
        for d in OPC_DETS:
            c = C[(C.site == site) & (C.severity == sev0) & (C.det == d)].sort_values("mult")
            a.plot(c.fpr.clip(lower=1e-4) * 100, c.event_recall,
                   marker="o", ms=3, lw=2.4 if d == "v2_full" else 1.1,
                   ls="--" if d in ("v2_no_peer", "v2_no_stateful") else "-",
                   color=COLORS.get(d, "#fd8d3c" if d == "v2_no_peer" else "#fdd0a2"), label=LABEL[d])
        a.set_xscale("log")
        a.set_xlabel("clean rows raising an alert, % (log)")
        a.set_title(f"{site}: operating curves")
    ax[0].set_ylabel("event recall")
    legend_below(fig, ax[1], ncol=4)
    save(fig, "fig8_operating_curves")

    # 7 worked examples
    exs = [(t, ex[t]) for t in ("spoof", "poison", "drift") if t in ex][:2]
    if exs:
        fig, axes = plt.subplots(3, len(exs), figsize=(5.4 * len(exs), 6), sharex="col", squeeze=False)
        for j, (t, (u, d)) in enumerate(exs):
            ev = [e for e in d["events"] if e[0] == t][0]
            _, s, e = ev
            lo, hi = max(s - 3 * DAY, 0), min(e + 3 * DAY, len(d["X"]))
            idx = d["X"].index[lo:hi]
            chg = {c: np.nansum(np.abs(d["X"][c].values[s:e] - d["df"][c].values[s:e])) / max(d["df"][c].std(), 1e-9)
                   for c in CH}
            ch = max(chg, key=chg.get)
            axes[0, j].plot(idx, d["df"][ch].iloc[lo:hi], lw=0.8, label="original")
            axes[0, j].plot(idx, d["X"][ch].iloc[lo:hi], lw=0.8, label="corrupted")
            axes[0, j].set_ylabel({"WC": "WC (m$^3$/m$^3$)", "ST": "ST (°C)", "SC": "SC (dS/m)"}[ch])
            axes[0, j].set_title(f"{t} event on {u}")
            axes[0, j].legend(fontsize=7)
            axes[1, j].plot(idx, d["norm"]["zscore"].iloc[lo:hi], lw=0.8, color="#9e9e9e", label="Rolling z-score")
            axes[1, j].plot(idx, d["norm"]["rules"].iloc[lo:hi], lw=0.8, color="#6baed6", label="Cross-sensor rules")
            axes[1, j].set_ylabel("score / threshold")
            axes[1, j].set_ylim(0, 4)
            axes[1, j].legend(fontsize=7)
            axes[2, j].plot(idx, d["norm"]["T1_peer"].iloc[lo:hi], lw=0.8, color="#fdae6b", label="Peer residual (v2)")
            axes[2, j].plot(idx, d["norm"]["T2_window"].iloc[lo:hi], lw=0.9, color="#e6550d", label="Windowed evidence (v2)")
            axes[2, j].set_ylim(0, 4)
            axes[2, j].set_ylabel("score / threshold")
            axes[2, j].legend(fontsize=7)
            for r in range(3):
                axes[r, j].axhline(1.0, color="r", ls="--", lw=0.7) if r > 0 else None
                axes[r, j].axvspan(d["X"].index[s], d["X"].index[min(e, len(d["X"]) - 1)], color="orange", alpha=0.18)
        import matplotlib.dates as mdates
        for j in range(len(exs)):
            loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
            axes[2, j].xaxis.set_major_locator(loc)
            axes[2, j].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        save(fig, "fig7_examples")


def make_matched_figures(MS, MT, out, OR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    sites = [x for x in ["Metompkin", "Crisfield"] if x in MS.site.unique()]
    sev0 = 1.0 if 1.0 in MS.severity.unique() else MS.severity.unique()[0]
    M1 = MS[MS.severity == sev0]
    dets = MAIN
    fig, ax = plt.subplots(1, len(sites), figsize=(5.2 * len(sites), 3.5), sharey=True, squeeze=False)
    w = 0.8 / len(dets)
    for a, site in zip(ax[0], sites):
        for i, d in enumerate(dets):
            v = M1[(M1.site == site) & (M1.det == d)].sort_values("target")
            a.bar(np.arange(len(TARGETS)) + (i - len(dets) / 2 + 0.5) * w, v.event_recall.values, w,
                  color=COLORS[d], label=LABEL[d], edgecolor="k", linewidth=0.3)
        a.set_xticks(range(len(TARGETS)))
        a.set_xticklabels([f"{t * 100:g}%" for t in TARGETS])
        a.set_xlabel("target false-alarm rate on clean validation data")
        a.set_title(f"{site}: event recall at matched alarms")
    ax[0][0].set_ylabel("event recall")
    legend_below(fig, ax[0][-1], ncol=3)
    finish_layout(fig)
    fig.savefig(out / "fig9_matched_recall.png", dpi=220)
    fig.savefig(out / "fig9_matched_recall.pdf")
    plt.close(fig)

    O1 = OR[OR.severity == sev0]
    fig, ax = plt.subplots(1, len(sites), figsize=(5.2 * len(sites), 3.5), sharey=True, squeeze=False)
    for a, site in zip(ax[0], sites):
        for i, d in enumerate(dets):
            v = O1[(O1.site == site) & (O1.det == d)].sort_values("target_fpr")
            a.bar(np.arange(len(TARGETS)) + (i - len(dets) / 2 + 0.5) * w, v.event_recall.values, w,
                  color=COLORS[d], label=LABEL[d], edgecolor="k", linewidth=0.3)
        a.set_xticks(range(len(TARGETS)))
        a.set_xticklabels([f"{t * 100:g}%" for t in TARGETS])
        a.set_xlabel("realised false-alarm rate on clean test rows")
        a.set_title(f"{site}: event recall at fixed test FPR")
    ax[0][0].set_ylabel("event recall")
    legend_below(fig, ax[0][-1], ncol=3)
    finish_layout(fig)
    fig.savefig(out / "fig11_recall_at_fixed_test_fpr.png", dpi=220)
    fig.savefig(out / "fig11_recall_at_fixed_test_fpr.pdf")
    plt.close(fig)

    cmp_dets = ["autoenc", "iforest", "hybrid", "v2_full", "t34_autoenc", "t34_iforest", "t34_ae_if", "v2_plus_autoenc"]
    fig, ax = plt.subplots(1, len(sites), figsize=(5.6 * len(sites), 3.6), sharey=True, squeeze=False)
    w2 = 0.8 / len(cmp_dets)
    for a, site in zip(ax[0], sites):
        for i, d in enumerate(cmp_dets):
            v = O1[(O1.site == site) & (O1.det == d)].sort_values("target_fpr")
            a.bar(np.arange(len(TARGETS)) + (i - len(cmp_dets) / 2 + 0.5) * w2, v.event_recall.values, w2,
                  color=COLORS[d], label=LABEL[d], edgecolor="k", linewidth=0.3)
        a.set_xticks(range(len(TARGETS)))
        a.set_xticklabels([f"{t * 100:g}%" for t in TARGETS])
        a.set_xlabel("realised false-alarm rate on clean test rows")
        a.set_title(f"{site}: specialised checks + generic ML")
    ax[0][0].set_ylabel("event recall")
    legend_below(fig, ax[0][-1], ncol=3)
    finish_layout(fig)
    fig.savefig(out / "fig12_combinations.png", dpi=220)
    fig.savefig(out / "fig12_combinations.pdf")
    plt.close(fig)

    tg = 0.02
    T1 = MT[(MT.severity == sev0) & (MT.target == tg)]
    fig, ax = plt.subplots(1, len(sites), figsize=(5.3 * len(sites), 3.4), sharey=True, squeeze=False)
    for a, site in zip(ax[0], sites):
        H = T1[T1.site == site].set_index("det").loc[dets, sa.ETYPES]
        im = a.imshow(H.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        a.set_xticks(range(len(sa.ETYPES)))
        a.set_xticklabels(sa.ETYPES)
        a.set_yticks(range(len(dets)))
        a.set_yticklabels([LABEL[d] for d in dets])
        a.set_title(f"{site}: recall by type at {tg * 100:g}% target FPR")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                a.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if H.values[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax[0].tolist(), fraction=0.025)
    fig.savefig(out / "fig10_matched_recall_by_type.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "fig10_matched_recall_by_type.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--met", required=True)
    ap.add_argument("--cris", required=True)
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--severities", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--n-per-type", type=int, default=6)
    ap.add_argument("--quantile", type=float, default=0.99, help="v1 baselines per-detector calibration quantile")
    ap.add_argument("--persist", type=int, default=2, help="v1 baselines persistence")
    ap.add_argument("--exclude-met", nargs="*", default=["VFMS-SP-S"])
    ap.add_argument("--roll-days", type=float, default=14.0, help="local baseline window in days")
    ap.add_argument("--only-site", default=None, choices=["Metompkin", "Crisfield"])
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True, parents=True)
    T0 = time.time()
    global ROLL2
    ROLL2 = int(a.roll_days * DAY)

    sites = {"Metompkin": sa.load_site(a.met, a.exclude_met), "Crisfield": sa.load_site(a.cris)}
    q_tier = 1 - (1 - a.quantile) / 4.0          # the four core tiers share the false-alarm budget
    ctx, refs, urs, thr1, thr2 = {}, {}, {}, {}, {}
    Ztrain = []
    for site, units in sites.items():
        ctx[site] = SiteCtx(units)
        refs[site], urs[site], thr1[site], thr2[site] = {}, {}, {}, {}
        for u in units:
            p = sa.partner_of(u, units)
            refs[site][u] = sa.UnitRef(units[u], units[p] if p else None)
            urs[site][u] = UnitV2(u, ctx[site], refs[site][u])
    for u, df in sites["Metompkin"].items():
        p = sa.partner_of(u, sites["Metompkin"])
        Z = refs["Metompkin"][u].z(df, sites["Metompkin"][p] if p else None)
        Ztrain.append(Z.iloc[refs["Metompkin"][u].tr])
    M = sa.fit_models(pd.concat(Ztrain), seed=0)
    print(f"[setup] calibration done in {time.time() - T0:.1f}s")

    for site, units in sites.items():            # clean-validation thresholds per unit
        for u, df in units.items():
            p = sa.partner_of(u, units)
            r = refs[site][u]
            S = sa.score(r.z(df, units[p] if p else None).iloc[r.va], M)
            thr1[site][u] = sa.thresholds(S, a.quantile)
            Tv = urs[site][u].tiers(df).iloc[r.va]
            th = {}
            for t in TIERS:
                if t == "T3_replay":
                    th[t] = 0.5
                elif t == "T4_flat":
                    th[t] = max(float(np.nanquantile(Tv[t], q_tier)), 1.0)
                else:
                    th[t] = max(float(np.nanquantile(Tv[t], q_tier)), 1e-6)
            thr2[site][u] = th
    print(f"[setup] thresholds done in {time.time() - T0:.1f}s")

    mstar = {}                                   # val-calibrated multipliers per unit / detector / target
    for site, units in sites.items():
        mstar[site] = {}
        for u, df in units.items():
            p = sa.partner_of(u, units)
            normc, _ = compute_norm(df, units[p] if p else None, refs[site][u], urs[site][u], M,
                                    thr1[site][u], thr2[site][u])
            mstar[site][u] = calibrate_mstar(normc, refs[site][u], a.persist)
    print(f"[setup] matched-FPR calibration done in {time.time() - T0:.1f}s")

    summ, typ, ex, evs, cvs, msum, mtyp = [], [], {}, [], [], [], []
    for sev in a.severities:
        for site, units in sites.items():
            if a.only_site and site != a.only_site:
                continue
            rows, evrows, pools, curves, mrows, mfa = [], [], [], [], [], []
            for sd in range(a.seeds):
                for u in units:
                    cap = (site == "Metompkin" and sev == 1.0 and sd == 0)
                    r, e, pl, cv, mr, mf, extra = evaluate_unit(
                        u, site, ctx[site], refs[site][u], urs[site][u], M, thr1[site][u], thr2[site][u],
                        mstar[site][u], 1000 * sd + zlib.crc32(u.encode()) % 997, sev, a.n_per_type,
                        a.persist, cap)
                    rows += r; evrows += e; pools.append(pl); curves += cv; mrows += mr; mfa += mf
                    if extra:
                        for t in ("spoof", "poison", "drift"):
                            if t not in ex and any(ev[0] == t for ev in extra["events"]):
                                ex[t] = (u, extra)
                print(f"[{site} sev {sev}] seed {sd + 1}/{a.seeds} done, elapsed {time.time() - T0:.0f}s", flush=True)
            s, t = summarize(rows, evrows, pools, sev, site)
            ms, mt = summarize_matched(mrows, mfa, sev, site)
            msum.append(ms)
            mtyp.append(mt)
            summ.append(s)
            typ.append(t)
            evs.append(pd.DataFrame(evrows).assign(site=site, severity=sev))
            cvs.append(pd.DataFrame(curves).assign(site=site, severity=sev))
            show = s.set_index("det").loc[MAIN + list(VARIANTS)[1:]]
            print(f"\n=== {site}, severity {sev} ===")
            print(show[["precision", "recall_rows", "fpr", "false_alerts_per_30d_per_unit", "event_recall",
                        "median_delay_min", "mean_undetected_exposure", "auroc", "avg_precision"]].round(3).to_string())
            if sev == 1.0:
                print(f"--- matched clean-validation FPR targets ({site}, severity 1.0): event recall / realised FPR ---")
                piv = ms.pivot(index="det", columns="target", values="event_recall").loc[OPC_DETS]
                pfp = ms.pivot(index="det", columns="target", values="realised_fpr").loc[OPC_DETS]
                print(piv.round(2).to_string())
                print(pfp.round(3).to_string())
    S, B = pd.concat(summ), pd.concat(typ)
    S.to_csv(out / "summary_v2.csv", index=False)
    B.to_csv(out / "recall_by_type_v2.csv", index=False)
    pd.concat(evs).to_csv(out / "events_all_v2.csv", index=False)
    C = pd.concat(cvs).groupby(["site", "severity", "det", "mult"]).agg(
        eps=("eps", "sum"), days=("clean_days", "sum"), n_ev=("n_ev", "sum"), n_det=("n_det", "sum"),
        fp=("fp", "sum"), neg=("neg", "sum")).reset_index()
    C["fpr"] = C.fp / C.neg
    C["false_alerts_per_30d_per_unit"] = C.eps / C.days * 30
    C["event_recall"] = C.n_det / C.n_ev
    C.to_csv(out / "operating_curves_v2.csv", index=False)
    OR = oracle_table(C)
    OR.to_csv(out / "recall_at_fixed_test_fpr.csv", index=False)
    if 1.0 in OR.severity.unique():
        print("\n=== event recall at fixed realised test FPR (severity 1.0) ===")
        print(OR[OR.severity == 1.0].pivot_table(index=["site", "det"], columns="target_fpr",
                                                values="event_recall").loc[
              lambda d: d.index.get_level_values("det").isin(OPC_DETS)].round(2).to_string())
    MS, MT = pd.concat(msum), pd.concat(mtyp)
    MS.to_csv(out / "matched_fpr_summary.csv", index=False)
    MT.to_csv(out / "matched_fpr_by_type.csv", index=False)

    # edge cost of the v2 tiers (microseconds per row, single core)
    site0 = sites["Metompkin"]
    u0 = next(iter(site0))
    t0 = time.time()
    urs["Metompkin"][u0].tiers(site0[u0])
    cost_us = (time.time() - t0) / len(site0[u0]) * 1e6
    json.dump(dict(v2_all_tiers_us_per_row_batch=cost_us, q_tier=q_tier, python=platform.python_version(),
                   machine=platform.platform(), args=vars(a), runtime_s=time.time() - T0),
              open(out / "run_info.json", "w"), indent=2)
    if not a.no_figs:
        make_figures(S, B, ex, out, C)
        make_matched_figures(MS, MT, out, OR)
    print(f"\nDone in {time.time() - T0:.0f}s. Send back the files in: {out.resolve()}")


if __name__ == "__main__":
    main()
