#!/usr/bin/env python3
# =============================================================================
# paper_figures.py  --  all paper figures + statistics from the saved results
# -----------------------------------------------------------------------------
# Reads the output folder of kaggle_merged_training.py (no GPU, no training):
#     !python paper_figures.py /kaggle/working/outputs_merged
# Writes <output folder>/figures/ :
#   fig1_roc_comparison        ROC: proposed vs all baselines vs paper model
#   fig2_pr_comparison         precision-recall curves
#   fig3_auc_forest            AUC with 95% CI for every model
#   fig4_bootstrap             bootstrap AUC + paired dAUC vs the paper model
#   fig5_perfold_auc           per-fold AUC: proposed vs paper vs BraTS-only control
#   fig6_confusion_matrix      proposed model at p = 0.5 (calibrated)
#   fig7_external_roc          leave-one-site-out ROC (UPenn, UCSF)
#   fig8_calibration           reliability diagram + Brier score
#   fig9_decision_curve        net benefit (decision curve analysis)
#   fig10_selective            accuracy vs coverage (confidence-based referral)
#   fig11_ablation             ablation dAUC (if ablation_table.csv exists)
#   table_comparison_holm.csv  DeLong vs proposed, Holm-adjusted across baselines
#   table_clinical_metrics.csv sensitivity/specificity/PPV/NPV/F1/MCC/Brier/ECE
#   table_efficiency.csv       parameters per network and per patient
#   table_selective.csv        accuracy at 100/50/30/20 % coverage
#   paper_numbers.txt          every key number in one place
# Each figure is saved as .png (200 dpi) and .pdf. A failing figure is reported
# and skipped; the others are still produced.
# =============================================================================
import os, re, sys, glob, json, traceback, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, matthews_corrcoef, f1_score)
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm, rankdata

OUT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/outputs_merged"
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
SEED, B = 42, 2000
PRIMARY_COLOR, PAPER_COLOR = "#1f3a8a", "#6b7280"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
numbers = []          # lines for paper_numbers.txt


def note(line):
    print(line); numbers.append(line)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("  saved", name)


# ---------------------------------------------------------------- statistics
def delong(y, p1, p2):
    y = np.asarray(y).astype(int); pos, neg = y == 1, y == 0
    def comps(p):
        d = p[pos][:, None] - p[neg][None, :]
        psi = (d > 0) + 0.5 * (d == 0)
        return psi.mean(1), psi.mean(0), psi.mean()
    a10, a01, a = comps(np.asarray(p1)); b10, b01, b = comps(np.asarray(p2))
    S = np.cov(np.vstack([a10, b10])) / pos.sum() + np.cov(np.vstack([a01, b01])) / neg.sum()
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0: return 0.0, 1.0
    z = (a - b) / np.sqrt(var); return float(z), float(2 * norm.sf(abs(z)))


def boot_auc(y, p, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) > 1: out.append(roc_auc_score(y[i], p[i]))
    return np.array(out)


def ci(y, p):
    b = boot_auc(y, p); return roc_auc_score(y, p), np.percentile(b, 2.5), np.percentile(b, 97.5)


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def calibrate(score, y, fold):
    """Cross-fitted Platt scaling: each fold is calibrated on the other folds only."""
    prob = np.zeros(len(y))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        lr = LogisticRegression(C=1e6).fit(score[tr, None], y[tr])
        prob[te] = lr.predict_proba(score[te, None])[:, 1]
    return prob


def metrics_at(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(float(thr), 3), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "PPV": tp / max(tp + fp, 1), "NPV": tn / max(tn + fn, 1), "accuracy": (tp + tn) / len(y),
            "F1": f1_score(y, pred), "MCC": matthews_corrcoef(y, pred), "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def ece(y, prob, bins=10):
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(edges, prob, "right") - 1, 0, bins - 1)
    return float(sum(abs(prob[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


# ---------------------------------------------------------------- load results
res = pd.read_csv(os.path.join(OUT, "results_table.csv"))
models = {}
for name in res.Model:
    f = os.path.join(OUT, f"oof_merged_{slug(name)}.csv")
    if os.path.isfile(f): models[name] = pd.read_csv(f)
any_df = next(iter(models.values()))
y = any_df.MGMT_value.values.astype(int); fold = any_df.fold.values.astype(int)
P = {n: d.pred.values.astype(float) for n, d in models.items()}
primary = next(n for n in P if "primary" in n)
paper = next((n for n in P if n.startswith("Paper headline")), None)
control = next((n for n in P if n.startswith("Control B0")), None)
baselines = [n for n in P if n.startswith("Baseline")]
merged_single = [n for n in P if n.startswith("Merged B")]
short = lambda n: (n.replace("  <- primary", "").replace(" (BraTS + UPENN+UCSF)", "")
                   .replace("Baseline ", "").replace(" (BraTS only)", " (BraTS only)"))
PRIMARY_LABEL = "Proposed: multi-cohort EfficientNet B0+B2"
label = lambda n: PRIMARY_LABEL if n == primary else ("Paper model (B0, BraTS only)" if n == paper else short(n))
note(f"Patients: {len(y)} (methylated {y.sum()}, rate {y.mean():.3f}); models found: {len(P)}")

palette = plt.cm.tab10.colors
order = [primary] + baselines + ([paper] if paper else [])
colors = {n: palette[(i + 1) % 10] for i, n in enumerate(baselines)}
colors[primary] = PRIMARY_COLOR
if paper: colors[paper] = PAPER_COLOR


def run(name, fn):
    try:
        fn()
    except Exception:
        print(f"  FIGURE FAILED: {name}\n{traceback.format_exc()}")


# ---------------------------------------------------------------- fig 1 / 2
def fig_roc_pr():
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for n in order:
        fpr, tpr, _ = roc_curve(y, P[n]); a = roc_auc_score(y, P[n])
        ax.plot(fpr, tpr, color=colors[n], lw=2.6 if n == primary else 1.3,
                ls="--" if n == paper else "-", label=f"{label(n)} (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#9ca3af", lw=1)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("ROC curves (same 574 patients, 5-fold cross-validation)")
    ax.legend(fontsize=7, loc="lower right", frameon=False); save(fig, "fig1_roc_comparison")

    fig, ax = plt.subplots(figsize=(5.6, 5))
    for n in order:
        pr, rc, _ = precision_recall_curve(y, P[n]); ap = average_precision_score(y, P[n])
        ax.plot(rc, pr, color=colors[n], lw=2.6 if n == primary else 1.3,
                ls="--" if n == paper else "-", label=f"{label(n)} (AP {ap:.3f})")
    ax.axhline(y.mean(), ls=":", color="#9ca3af", lw=1, label=f"Prevalence ({y.mean():.2f})")
    ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-recall curves"); ax.legend(fontsize=7, loc="lower left", frameon=False)
    save(fig, "fig2_pr_comparison")


# ---------------------------------------------------------------- fig 3 + Holm table
def fig_forest_and_holm():
    rows, names = [], [primary] + baselines + [n for n in (control, paper) if n]
    for n in names:
        a, lo, hi = ci(y, P[n]); rows.append((n, a, lo, hi))
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(rows) + 1))
    for k, (n, a, lo, hi) in enumerate(rows[::-1]):
        c = colors.get(n, "#374151")
        ax.errorbar(a, k, xerr=[[a - lo], [hi - a]], fmt="o", color=c, ms=6 if n == primary else 4, capsize=3,
                    lw=2 if n == primary else 1.2)
        ax.text(hi + 0.005, k, f"{a:.3f} ({lo:.3f}-{hi:.3f})", va="center", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([label(n) for n, *_ in rows[::-1]], fontsize=8)
    ax.axvline(0.5, ls=":", color="#9ca3af"); ax.set_xlim(0.5, 0.86)
    ax.set_xlabel("AUC (95% bootstrap CI)"); ax.set_title("Discrimination of all models")
    save(fig, "fig3_auc_forest")

    comp = baselines + [n for n in (control, paper) if n]
    HOLM_COL = f"Holm-adjusted p ({len(baselines)} baselines)"
    zp = [delong(y, P[primary], P[n]) for n in comp]
    adj = holm([p for _, p in zp[:len(baselines)]]) if baselines else []
    out = []
    for k, n in enumerate(comp):
        a, lo, hi = ci(y, P[n]); z, p = zp[k]
        h = adj[k] if k < len(baselines) else np.nan
        out.append({"Model": label(n), "AUC (95% CI)": f"{a:.3f} ({lo:.3f}-{hi:.3f})",
                    "Proposed - model": f"{roc_auc_score(y, P[primary]) - a:+.3f}", "DeLong z": round(z, 2),
                    "DeLong p": round(p, 4),
                    HOLM_COL: "-" if np.isnan(h) else round(float(h), 4),
                    "Significant after Holm (0.05)": "-" if np.isnan(h) else ("yes" if h < 0.05 else "no")})
    t = pd.DataFrame(out); t.to_csv(os.path.join(FIG, "table_comparison_holm.csv"), index=False)
    print(t.to_string(index=False))
    for r in out: note(f"vs {r['Model']}: dAUC {r['Proposed - model']}, DeLong p {r['DeLong p']}, Holm {r[HOLM_COL]}")


# ---------------------------------------------------------------- fig 4
def fig_bootstrap():
    rng = np.random.default_rng(SEED); n = len(y); a_b, d_b = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2: continue
        a = roc_auc_score(y[i], P[primary][i]); a_b.append(a)
        if paper: d_b.append(a - roc_auc_score(y[i], P[paper][i]))
    a_b, d_b = np.array(a_b), np.array(d_b)
    fig, axes = plt.subplots(1, 2 if paper else 1, figsize=(10 if paper else 5.2, 3.8), squeeze=False)
    ax = axes[0, 0]; lo, hi = np.percentile(a_b, [2.5, 97.5])
    ax.hist(a_b, bins=40, color=PRIMARY_COLOR, alpha=0.8)
    ax.axvspan(lo, hi, color=PRIMARY_COLOR, alpha=0.08)
    if paper: ax.axvline(roc_auc_score(y, P[paper]), color=PAPER_COLOR, ls="--", label="Paper model AUC")
    if control: ax.axvline(roc_auc_score(y, P[control]), color="#b45309", ls=":", label="BraTS-only control")
    ax.set_title(f"Bootstrap AUC of the proposed model (B={len(a_b)})\n95% CI {lo:.3f}-{hi:.3f}")
    ax.set_xlabel("AUC"); ax.legend(fontsize=7, frameon=False)
    note(f"Proposed AUC bootstrap 95% CI {lo:.3f}-{hi:.3f}")
    if paper:
        ax = axes[0, 1]; lo, hi = np.percentile(d_b, [2.5, 97.5]); frac = (d_b > 0).mean()
        ax.hist(d_b, bins=40, color="#047857", alpha=0.8); ax.axvline(0, color="k", lw=1)
        ax.set_title(f"Paired bootstrap: proposed - paper model\n"
                     f"dAUC 95% CI {lo:+.3f} to {hi:+.3f}; P(dAUC > 0) = {frac:.3f}")
        ax.set_xlabel("dAUC")
        note(f"Paired dAUC vs paper model: 95% CI {lo:+.3f} to {hi:+.3f}, P(>0) = {frac:.3f}")
    save(fig, "fig4_bootstrap")


# ---------------------------------------------------------------- fig 5
def fig_perfold():
    names = [n for n in (primary, control, paper) if n]
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); w = 0.8 / len(names); folds = sorted(np.unique(fold))
    for k, n in enumerate(names):
        vals = [roc_auc_score(y[fold == f], P[n][fold == f]) for f in folds]
        ax.bar(np.arange(len(folds)) + k * w, vals, w, label=f"{label(n)} (mean {np.mean(vals):.3f})",
               color=colors.get(n, "#b45309"))
    ax.set_xticks(np.arange(len(folds)) + w * (len(names) - 1) / 2)
    ax.set_xticklabels([f"Fold {f+1}" for f in folds]); ax.set_ylim(0.5, 0.85); ax.set_ylabel("AUC")
    ax.set_title("Per-fold AUC on the held-out folds"); ax.legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "fig5_perfold_auc")


# ---------------------------------------------------------------- calibrated probabilities
PROB = None


def calibrated():
    global PROB
    if PROB is None: PROB = calibrate(P[primary], y, fold)
    return PROB


# ---------------------------------------------------------------- fig 6 + clinical table
def fig_confusion_and_clinical():
    prob = calibrated()
    m = metrics_at(y, prob, 0.5)
    fpr, tpr, thr = roc_curve(y, prob); spec = 1 - fpr
    k = np.where(spec >= 0.90)[0][-1]; m90 = metrics_at(y, prob, thr[k])
    brier = float(np.mean((prob - y) ** 2)); e = ece(y, prob)
    rows = [dict(operating_point="balanced (p >= 0.5)", **m), dict(operating_point="high specificity (>= 0.90)", **m90)]
    t = pd.DataFrame(rows)
    for c in ("sensitivity", "specificity", "PPV", "NPV", "accuracy", "F1", "MCC"): t[c] = t[c].round(3)
    t["Brier"] = round(brier, 3); t["ECE"] = round(e, 3)
    t.to_csv(os.path.join(FIG, "table_clinical_metrics.csv"), index=False); print(t.to_string(index=False))
    note(f"At p>=0.5: sens {m['sensitivity']:.3f}, spec {m['specificity']:.3f}, PPV {m['PPV']:.3f}, "
         f"NPV {m['NPV']:.3f}, acc {m['accuracy']:.3f}, MCC {m['MCC']:.3f}")
    note(f"At spec>=0.90: sens {m90['sensitivity']:.3f}, spec {m90['specificity']:.3f}, PPV {m90['PPV']:.3f}")
    note(f"Calibration: Brier {brier:.3f}, ECE {e:.3f}")
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4)); ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Unmethylated", "Methylated"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Unmethylated", "Methylated"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Proposed model, p >= 0.5\nsens {m['sensitivity']:.2f} | spec {m['specificity']:.2f} | "
                 f"acc {m['accuracy']:.2f}", fontsize=8)
    save(fig, "fig6_confusion_matrix")


# ---------------------------------------------------------------- fig 7
def fig_external():
    files = sorted(glob.glob(os.path.join(OUT, "external_preds_*.csv")))
    if not files: print("  no external predictions -> fig7 skipped"); return
    dd = os.path.join(OUT, "dedup_report.csv")
    dedup = pd.read_csv(dd) if os.path.isfile(dd) else None
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for f, c in zip(files, ["#1f3a8a", "#b91c1c", "#047857"]):
        site = os.path.basename(f).replace("external_preds_", "").replace(".csv", "")
        d = pd.read_csv(f); yy, pp = d.MGMT_value.values.astype(int), d.pred.values
        a, lo, hi = ci(yy, pp); fpr, tpr, _ = roc_curve(yy, pp)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{site} (n={len(yy)}) AUC {a:.3f} ({lo:.3f}-{hi:.3f})")
        note(f"External {site}: n={len(yy)}, AUC {a:.3f} ({lo:.3f}-{hi:.3f})")
        if dedup is not None:
            near = set(dedup.loc[(dedup.site == site) & (~dedup.excluded.astype(bool)) & (dedup.max_cosine > 0.90),
                                 "subject_id"])
            keep = ~d.subject_id.isin(near).values
            if 0 < (~keep).sum() and len(np.unique(yy[keep])) > 1:
                a2, lo2, hi2 = ci(yy[keep], pp[keep]); fpr, tpr, _ = roc_curve(yy[keep], pp[keep])
                ax.plot(fpr, tpr, color=c, lw=1.4, ls="--",
                        label=f"{site} excl. {int((~keep).sum())} borderline (n={int(keep.sum())}) AUC {a2:.3f}")
                note(f"External {site} excluding borderline (sim 0.90-0.95): n={int(keep.sum())}, "
                     f"AUC {a2:.3f} ({lo2:.3f}-{hi2:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#9ca3af")
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Leave-one-site-out external validation\n(model never trained on the held-out site)")
    ax.legend(fontsize=7, loc="lower right", frameon=False); save(fig, "fig7_external_roc")


# ---------------------------------------------------------------- fig 8
def fig_calibration():
    prob = calibrated(); bins = 10
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, prob, "right") - 1, 0, bins - 1)
    mp = [prob[idx == b].mean() for b in range(bins) if (idx == b).any()]
    fo = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], ":", color="#9ca3af", label="Perfect calibration")
    ax.plot(mp, fo, "o-", color=PRIMARY_COLOR, label="Proposed model")
    ax.set_ylabel("Observed methylation rate")
    ax.set_title(f"Calibration (cross-fitted Platt scaling)\nBrier {np.mean((prob - y) ** 2):.3f} | "
                 f"ECE {ece(y, prob):.3f}")
    ax.legend(fontsize=7, frameon=False)
    ax2.hist(prob, bins=20, color=PRIMARY_COLOR, alpha=0.7); ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Patients"); save(fig, "fig8_calibration")


# ---------------------------------------------------------------- fig 9
def fig_decision_curve():
    prob = calibrated(); n = len(y); prev = y.mean(); th = np.linspace(0.05, 0.95, 91)
    nb_model, nb_all = [], []
    for t in th:
        pred = prob >= t; tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        nb_model.append(tp / n - fp / n * t / (1 - t)); nb_all.append(prev - (1 - prev) * t / (1 - t))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(th, nb_model, color=PRIMARY_COLOR, lw=2, label="Proposed model")
    ax.plot(th, nb_all, color="#b45309", lw=1.2, label="Treat all as methylated")
    ax.axhline(0, color="k", lw=1, label="Treat none")
    ax.set_ylim(-0.05, max(max(nb_model), prev) + 0.05)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis"); ax.legend(fontsize=7, frameon=False)
    better = th[np.array(nb_model) > np.maximum(np.array(nb_all), 0) + 1e-9]
    if len(better): note(f"Decision curve: model has higher net benefit than treat-all/none for thresholds "
                         f"{better.min():.2f}-{better.max():.2f}")
    save(fig, "fig9_decision_curve")


# ---------------------------------------------------------------- fig 10
def fig_selective():
    prob = calibrated(); conf = np.abs(prob - 0.5); order_ = np.argsort(-conf); pred = (prob >= 0.5).astype(int)
    cov = np.linspace(0.1, 1.0, 46); acc = []
    for c in cov:
        k = max(1, int(round(c * len(y)))); i = order_[:k]; acc.append((pred[i] == y[i]).mean())
    rows = []
    for c in (1.0, 0.5, 0.3, 0.2):
        k = int(round(c * len(y))); i = order_[:k]
        rows.append({"coverage": f"{int(c*100)}%", "patients": k, "accuracy": round(float((pred[i] == y[i]).mean()), 3)})
    pd.DataFrame(rows).to_csv(os.path.join(FIG, "table_selective.csv"), index=False)
    for r in rows: note(f"Selective prediction: coverage {r['coverage']} ({r['patients']} pts) accuracy {r['accuracy']}")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(cov * 100, acc, color=PRIMARY_COLOR, lw=2)
    for r in rows[1:]:
        ax.plot(int(r["coverage"][:-1]), r["accuracy"], "o", color="#b91c1c")
        ax.annotate(f"{r['accuracy']:.2f}", (int(r["coverage"][:-1]), r["accuracy"]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Coverage: % of patients with the most confident predictions"); ax.set_ylabel("Accuracy")
    ax.invert_xaxis(); ax.set_title("Confidence-based selective prediction")
    save(fig, "fig10_selective")


# ---------------------------------------------------------------- fig 11
def fig_ablation():
    f = os.path.join(OUT, "ablation_table.csv")
    if not os.path.isfile(f): print("  no ablation_table.csv -> fig11 skipped"); return
    t = pd.read_csv(f); t = t[t.dAUC.astype(str) != "-"].copy()
    t["d"] = t.dAUC.astype(float); t["p"] = t["DeLong p vs full"].astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(t) + 1))
    cols = ["#b91c1c" if p < 0.05 else "#9ca3af" for p in t.p]
    ax.barh(range(len(t)), t.d, color=cols)
    for k, (d, p) in enumerate(zip(t.d, t.p)):
        ax.text(d + (-0.002 if d < 0 else 0.002), k, f"{d:+.3f} (p={p:.3f})", va="center",
                ha="right" if d < 0 else "left", fontsize=7)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.Variant, fontsize=8); ax.axvline(0, color="k", lw=1)
    lo = min(t.d.min(), 0) - 0.03; hi = max(t.d.max(), 0) + 0.03; ax.set_xlim(lo, hi)
    ax.set_xlabel("Change in AUC when the component is removed (red: DeLong p < 0.05)")
    ax.set_title("Ablation of the proposed model"); ax.invert_yaxis(); save(fig, "fig11_ablation")
    for _, r in t.iterrows(): note(f"Ablation {r.Variant}: dAUC {r.d:+.3f}, p {r.p:.4f}, strict dAUC {r['dAUC strict']}")


# ---------------------------------------------------------------- efficiency table
def table_efficiency():
    try:
        import torch.nn as nn, timm
    except Exception:
        print("  torch/timm not available -> efficiency table skipped"); return
    def count(name):
        if name == "simple_cnn":
            blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            m = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256), nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(), nn.Linear(256, 1))
        else:
            m = timm.create_model(name, pretrained=False, num_classes=1)
        return sum(p.numel() for p in m.parameters()) / 1e6
    known = [("VGG-16", "vgg16_bn"), ("DenseNet-121", "densenet121"), ("ResNet-18", "resnet18"),
             ("MobileNet-v2", "mobilenetv2_100"), ("MobileNet-v3-Large", "mobilenetv3_large_100"),
             ("Simple CNN", "simple_cnn")]
    # only the baselines that are in this run's results table
    specs = [("Proposed (EfficientNet-B0 + B2)", ["efficientnet_b0", "efficientnet_b2"])] + \
            [(lab, [net]) for lab, net in known if any(n.startswith(f"Baseline {lab}") for n in baselines)]
    rows = []
    for lab, nets in specs:
        per = sum(count(n) for n in nets)
        rows.append({"Model": lab, "Networks per MRI type": len(nets), "Params per MRI type (M)": round(per, 2),
                     "Networks per patient (4 MRI types)": 4 * len(nets), "Total params per patient (M)": round(4 * per, 1)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(FIG, "table_efficiency.csv"), index=False)
    print(t.to_string(index=False))
    for r in rows: note(f"Params {r['Model']}: {r['Total params per patient (M)']} M total")


for name, fn in [("ROC/PR", fig_roc_pr), ("forest + Holm", fig_forest_and_holm), ("bootstrap", fig_bootstrap),
                 ("per-fold", fig_perfold), ("confusion + clinical", fig_confusion_and_clinical),
                 ("external", fig_external), ("calibration", fig_calibration),
                 ("decision curve", fig_decision_curve), ("selective", fig_selective),
                 ("ablation", fig_ablation), ("efficiency", table_efficiency)]:
    print(f"\n--- {name}")
    run(name, fn)

gc = [f for f in glob.glob(os.path.join(OUT, "fig_gradcam.*"))]
for f in gc:
    import shutil; shutil.copy(f, os.path.join(FIG, os.path.basename(f).replace("fig_gradcam", "fig12_gradcam")))
if gc: print("\n--- Grad-CAM figure copied as fig12_gradcam")

with open(os.path.join(FIG, "paper_numbers.txt"), "w") as fh:
    fh.write("\n".join(numbers) + "\n")
print(f"\nAll figures and tables are in {FIG}")
