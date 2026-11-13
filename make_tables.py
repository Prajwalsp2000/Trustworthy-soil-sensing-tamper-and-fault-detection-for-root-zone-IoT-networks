#!/usr/bin/env python3
"""Builds tables_auto.tex for the chapter from the result CSVs (default folder: results_v2c)."""
import sys
from pathlib import Path
import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v2c")
O = pd.read_csv(R / "recall_at_fixed_test_fpr.csv")
S = pd.read_csv(R / "summary_v2.csv")
MS = pd.read_csv(R / "matched_fpr_summary.csv")
MT = pd.read_csv(R / "matched_fpr_by_type.csv")
NAME = {"zscore": "Rolling z-score", "rules": "Cross-sensor rules", "iforest": "Isolation Forest",
        "autoenc": "Autoencoder", "hybrid": "Rules + Isolation Forest",
        "t34_autoenc": "Replay + flat-run + autoencoder", "t34_iforest": "Replay + flat-run + Isolation Forest",
        "v2_full": "v2 (peer, windowed, replay, flat-run)", "v2_plus_autoenc": "v2 + autoencoder",
        "v2_no_peer": "v2 without peer regression", "v2_no_stateful": "v2 without windowed evidence",
        "v2_with_slow": "v2 + slow depth-gap tier"}
SITES = ["Metompkin", "Crisfield"]
TG = [0.005, 0.01, 0.02, 0.04]
out = []


def best_mark(vals, v):
    return f"\\textbf{{{v:.2f}}}" if abs(v - max(vals)) < 1e-9 else f"{v:.2f}"


def pick(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        if k in ("target", "target_fpr"):
            m &= (df[k] - v).abs() < 1e-9
        else:
            m &= df[k] == v
    return df[m].iloc[0]


# ---- recall at fixed test FPR (severity 1)
dets = ["zscore", "rules", "iforest", "autoenc", "hybrid", "t34_iforest", "t34_autoenc", "v2_full", "v2_plus_autoenc"]
o1 = O[O.severity == 1.0]
t = ["\\begin{table}[t]\\centering\\small",
     "\\caption{Event recall at a fixed realised false-alarm rate on clean test rows (severity multiplier 1.0; mean over 3 seeds). Best value in each column in bold.}",
     "\\label{tab:fixedfpr}",
     "\\begin{tabular}{l" + "cccc" * 2 + "}\\toprule",
     " & \\multicolumn{4}{c}{Metompkin (in-site)} & \\multicolumn{4}{c}{Crisfield (held-out)}\\\\",
     "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
     "Detector & 0.5\\% & 1\\% & 2\\% & 4\\% & 0.5\\% & 1\\% & 2\\% & 4\\%\\\\\\midrule"]
col = {}
for s in SITES:
    for g in TG:
        col[(s, g)] = {d: float(pick(o1, site=s, det=d, target_fpr=g).event_recall) for d in dets}
for d in dets:
    cells = [best_mark(list(col[(s, g)].values()), col[(s, g)][d]) for s in SITES for g in TG]
    t.append(f"{NAME[d]} & " + " & ".join(cells) + "\\\\")
    if d == "hybrid":
        t.append("\\midrule")
t += ["\\bottomrule\\end{tabular}\\end{table}"]
out += t

# ---- severity sensitivity at 2% FPR
dets2 = ["autoenc", "hybrid", "t34_autoenc", "v2_full", "v2_plus_autoenc"]
t = ["\\begin{table}[t]\\centering\\small",
     "\\caption{Event recall at a fixed 2\\% realised false-alarm rate for three severity multipliers.}",
     "\\label{tab:severity}", "\\begin{tabular}{l" + "ccc" * 2 + "}\\toprule",
     " & \\multicolumn{3}{c}{Metompkin} & \\multicolumn{3}{c}{Crisfield}\\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Detector & 0.5 & 1.0 & 2.0 & 0.5 & 1.0 & 2.0\\\\\\midrule"]
for d in dets2:
    cells = [f"{float(pick(O, site=s, det=d, severity=sev, target_fpr=0.02).event_recall):.2f}"
             for s in SITES for sev in (0.5, 1.0, 2.0)]
    t.append(f"{NAME[d]} & " + " & ".join(cells) + "\\\\")
t += ["\\bottomrule\\end{tabular}\\end{table}"]
out += t

# ---- by event type, validation-calibrated 1% target
dets3 = ["autoenc", "hybrid", "t34_autoenc", "v2_full"]
ety = ["spike", "stuck", "drift", "spoof", "replay", "poison"]
t = ["\\begin{table}[t]\\centering\\small",
     "\\caption{Event recall by event type when every detector's threshold is calibrated to a 1\\% false-alarm target on clean validation data (severity 1.0). The realised false-alarm rate on the test period (FPR) differs between detectors and should be read together with recall.}",
     "\\label{tab:bytype}", "\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llc" + "c" * 6 + "}\\toprule",
     "Site & Detector & FPR & Spike & Stuck & Drift & Spoof & Replay & Poison\\\\\\midrule"]
for s in SITES:
    for d in dets3:
        r = pick(MT, site=s, det=d, severity=1.0, target=0.01)
        f = float(pick(MS, site=s, det=d, severity=1.0, target=0.01).realised_fpr)
        t.append(f"{s if d == dets3[0] else ''} & {NAME[d]} & {100 * f:.1f}\\% & " + " & ".join(f"{r[e]:.2f}" for e in ety) + "\\\\")
    if s != SITES[-1]:
        t.append("\\midrule")
t += ["\\bottomrule\\end{tabular}\\end{table}"]
out += t

# ---- calibration drift
dets4 = ["iforest", "autoenc", "hybrid", "t34_autoenc", "v2_full"]
t = ["\\begin{table}[t]\\centering\\small",
     "\\caption{Calibration drift: realised false-alarm rate (\\%) on the test period when thresholds are set on clean validation data to reach the target in the header (severity 1.0).}",
     "\\label{tab:drift}", "\\begin{tabular}{l" + "cccc" * 2 + "}\\toprule",
     " & \\multicolumn{4}{c}{Metompkin} & \\multicolumn{4}{c}{Crisfield}\\\\",
     "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}", "Detector & 0.5\\% & 1\\% & 2\\% & 4\\% & 0.5\\% & 1\\% & 2\\% & 4\\%\\\\\\midrule"]
for d in dets4:
    cells = [f"{100 * float(pick(MS, site=s, det=d, severity=1.0, target=g).realised_fpr):.1f}" for s in SITES for g in TG]
    t.append(f"{NAME[d]} & " + " & ".join(cells) + "\\\\")
t += ["\\bottomrule\\end{tabular}\\end{table}"]
out += t

# ---- ablation at fixed FPR
dets5 = ["t34_autoenc", "v2_full", "v2_no_peer", "v2_no_stateful", "v2_with_slow", "v2_plus_autoenc"]
t = ["\\begin{table}[t]\\centering\\small",
     "\\caption{Component ablation: event recall at fixed realised false-alarm rates of 1\\% and 2\\% (severity 1.0).}",
     "\\label{tab:ablation}", "\\begin{tabular}{lcccc}\\toprule",
     " & \\multicolumn{2}{c}{Metompkin} & \\multicolumn{2}{c}{Crisfield}\\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Variant & 1\\% & 2\\% & 1\\% & 2\\%\\\\\\midrule"]
for d in dets5:
    cells = [f"{float(pick(o1, site=s, det=d, target_fpr=g).event_recall):.2f}" for s in SITES for g in (0.01, 0.02)]
    t.append(f"{NAME[d]} & " + " & ".join(cells) + "\\\\")
t += ["\\bottomrule\\end{tabular}\\end{table}"]
out += t

# ---- operating point, delay and exposure
dets6 = ["autoenc", "hybrid", "t34_autoenc", "v2_full"]
s1 = S[S.severity == 1.0]
t = ["\\begin{table}[t]\\centering\\small",
     "\\caption{Detectors at their default (validation-calibrated, threshold multiplier 1) operating points, severity 1.0: false alert episodes per sensor per 30 days, event recall, median detection delay in hours, and the mean share of attack-induced deviation that occurred before detection (undetected exposure; lower is better).}",
     "\\label{tab:operating}", "\\begin{tabular}{llcccc}\\toprule",
     "Site & Detector & False alerts & Recall & Delay (h) & Exposure\\\\\\midrule"]
for s in SITES:
    for d in dets6:
        r = pick(s1, site=s, det=d)
        t.append(f"{s if d == dets6[0] else ''} & {NAME[d]} & {r.false_alerts_per_30d_per_unit:.1f} & {r.event_recall:.2f} & {r.median_delay_min / 60:.1f} & {r.mean_undetected_exposure:.2f}\\\\")
    if s != SITES[-1]:
        t.append("\\midrule")
t += ["\\bottomrule\\end{tabular}\\end{table}"]
out += t

txt = "\n".join(out) + "\n"
blocks = ["\\begin{table}" + b for b in txt.split("\\begin{table}")[1:]]
order = ["tab:fixedfpr", "tab:severity", "tab:bytype", "tab:operating", "tab:ablation", "tab:drift"]
blocks.sort(key=lambda b: next(i for i, l in enumerate(order) if "\\label{" + l + "}" in b))
Path("tables_auto.tex").write_text("\n".join(blocks) + "\n")
print("wrote tables_auto.tex")
