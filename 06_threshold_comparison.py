import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
from scipy.stats import chi2 as chi2_dist, norm as scipy_norm
from sklearn.metrics import roc_auc_score


RAW_DATA = "python_data/labeled_data_with_types.csv"
CV_PREDS = "python_results/binary/cv_predictions.csv"
METRICS  = "python_results/binary/performance_metrics.json"
REF_DATA = "python_data/prob_ref.csv"
OUTDIR   = "python_results/threshold_comparison"
os.makedirs(OUTDIR, exist_ok=True)

THRESHOLDS = [95, 200, 400]

TX_MAP = {
    "all":    "All data",
    "500_1":  "500 (100%)",
    "400_1":  "400 (100%)",
    "400_07": "400 (70%)",
}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return round(p, 4), round(max(0, centre - margin), 4), round(min(1, centre + margin), 4)


def delong_test(y_true, score_a, score_b):
    """
    DeLong et al. (1988) non-parametric test for comparing two correlated AUCs.
    Uses placement values (Mann-Whitney kernel). Two-sided. Returns (z, p).

    Note: all three threshold methods share the same score (max_acc_g), so the
    DeLong result will be identical for Threshold_95g, _200g, and _400g.
    When both scores are identical, var_diff == 0 and (0.0, 1.0) is returned.
    """
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n1 = pos_mask.sum()
    n0 = neg_mask.sum()

    sa_pos, sa_neg = score_a[pos_mask], score_a[neg_mask]
    sb_pos, sb_neg = score_b[pos_mask], score_b[neg_mask]

    def pv_pos(s_pos, s_neg):
        return np.array([np.mean(s > s_neg) + 0.5 * np.mean(s == s_neg)
                         for s in s_pos])

    def pv_neg(s_pos, s_neg):
        return np.array([np.mean(s < s_pos) + 0.5 * np.mean(s == s_pos)
                         for s in s_neg])

    v10_a = pv_pos(sa_pos, sa_neg)
    v10_b = pv_pos(sb_pos, sb_neg)
    v01_a = pv_neg(sa_pos, sa_neg)
    v01_b = pv_neg(sb_pos, sb_neg)

    S10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    S01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)

    var_diff = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / n1 + \
               (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n0

    if var_diff <= 0:
        return 0.0, 1.0

    auc_a = v10_a.mean()
    auc_b = v10_b.mean()
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * scipy_norm.sf(abs(z)))
    return float(z), p


# ── Reference data — embedded table with collision + mortality ────────────────
# Columns: method_key, treatment, collision_pct, mortality_pct, n, k_collision
# k_collision = None → use pseudo-count from pct * n

_FULL_ROWS = [
    # total_obs: collision only
    ("total_obs",  "500 (100%)", 47.22, np.nan, 108, 51),
    ("total_obs",  "400 (100%)", 39.23, np.nan, 130, 51),
    ("total_obs",  "400 (70%)",  48.72, np.nan, 117, 57),
    # total_util: collision + mortality
    ("total_util", "500 (100%)", 50.00, 19.40,  76,  38),
    ("total_util", "400 (100%)", 39.22,  8.71, 102,  40),
    ("total_util", "400 (70%)",  50.00,  9.97,  80,  40),
    # van Esch & Spierts 2014
    ("esch_2014",  "500 (100%)", 38.66, 38.19,  76,  None),
    ("esch_2014",  "400 (100%)", 38.88, 26.25, 102,  None),
    ("esch_2014",  "400 (70%)",  55.93, 37.76,  80,  None),
    # EVS-EN 18110 2025
    ("cen_2025",   "500 (100%)", 45.33, 17.78,  76,  None),
    ("cen_2025",   "400 (100%)", 45.59, 10.25, 102,  None),
    ("cen_2025",   "400 (70%)",  65.57, 13.23,  80,  None),
    # MiniRocket
    ("MiniRocket", "500 (100%)", 51.31, 19.91,  76,  None),
    ("MiniRocket", "400 (100%)", 36.27,  8.06, 102,  None),
    ("MiniRocket", "400 (70%)",  55.00, 10.97,  80,  None),
]

# Excl. "Other impeller collision" subset
# total_util_leading n stays at total_util n (video-based);
# MiniRocket_leading n is the filtered sensor subset.
_EXCL_ROWS = [
    ("total_util_leading", "500 (100%)", 36.84, 14.29,  76, None),
    ("total_util_leading", "400 (100%)", 30.39,  6.75, 102, None),
    ("total_util_leading", "400 (70%)",  42.50,  8.48,  80, None),
    ("MiniRocket_leading", "500 (100%)", 45.45, 17.63,  66, None),
    ("MiniRocket_leading", "400 (100%)", 33.33,  7.41,  93, None),
    ("MiniRocket_leading", "400 (70%)",  52.70, 10.51,  74, None),
    # esch/cen: same predictions, CI denom = MiniRocket_leading n
    ("esch_2014",  "500 (100%)", 38.66, 38.19,  66, None),
    ("esch_2014",  "400 (100%)", 38.88, 26.25,  93, None),
    ("esch_2014",  "400 (70%)",  55.93, 37.76,  74, None),
    ("cen_2025",   "500 (100%)", 45.33, 17.78,  66, None),
    ("cen_2025",   "400 (100%)", 45.59, 10.25,  93, None),
    ("cen_2025",   "400 (70%)",  65.57, 13.23,  74, None),
]

def _build_ref_dict(rows):
    """Build {(method, tx): {collision, mortality, n, k_collision}} lookup."""
    d = {}
    for meth, tx, col_pct, mort_pct, n, k in rows:
        k_col = k if k is not None else round(col_pct / 100 * n)
        d[(meth, tx)] = {
            "collision": col_pct / 100,
            "mortality": mort_pct / 100 if not (isinstance(mort_pct, float) and np.isnan(mort_pct)) else np.nan,
            "n":         n,
            "k_col":     k_col,
        }
    return d

FULL_REF = _build_ref_dict(_FULL_ROWS)
EXCL_REF = _build_ref_dict(_EXCL_ROWS)

# Legacy helpers — used by CSV output section
def _ref_lookup(ref_dict, meth, tx, metric):
    """Return (p, lo, hi, n) from a ref dict, computing Wilson CI from pseudo-counts."""
    key = (meth, tx)
    if key not in ref_dict:
        return np.nan, np.nan, np.nan, 0
    row = ref_dict[key]
    p = row[metric]
    n = row["n"]
    if np.isnan(p) or n == 0:
        return np.nan, np.nan, np.nan, 0
    k = row["k_col"] if metric == "collision" else round(p * n)
    _, lo, hi = wilson_ci(k, n)
    return p, lo, hi, n

# Keep these for backward compat with prob_df lookups (threshold CIs)
def obs_ci_n(ref_dict_key, tx):
    return _ref_lookup(FULL_REF, ref_dict_key, tx, "collision")

def pred_ci_n(ref_df, tx, denom_ref=None):
    return np.nan, np.nan, np.nan, 0   # replaced by FULL_REF / EXCL_REF

# Thin wrappers used by CSV builder
total_obs_ref  = None
total_util_ref = None
esch_2014_ref  = None
cen_2025_ref   = None


# ── Load raw + CV predictions ─────────────────────────────────────────────────
raw = pd.read_csv(RAW_DATA, low_memory=False)

acc_max = (
    raw.groupby("file")["higacc_mag_g"]
    .max()
    .reset_index()
    .rename(columns={"higacc_mag_g": "max_acc_g"})
)
print(f"  Unique files after filter: {len(acc_max)}")

cv = pd.read_csv(CV_PREDS)
print(f"  Files in CV predictions: {len(cv)}")

df = cv.merge(acc_max, on="file", how="inner")
print(f"  Files matched after merge: {len(df)}")
if len(df) < len(cv):
    print(f"  WARNING: {len(cv) - len(df)} CV files not found in filtered raw data")

if "passage_type" not in df.columns:
    raise ValueError("passage_type column not found in cv_predictions.csv — required for secondary MiniRocket.")

y_true = df["y_true"].values

for thr in THRESHOLDS:
    df[f"pred_{thr}g"] = (df["max_acc_g"] >= thr).astype(int)

df.to_csv(f"{OUTDIR}/threshold_predictions.csv", index=False)


# ── Secondary MiniRocket: leading-edge only (exclude Other impeller collision) ─
df_leading = df[df["passage_type"] != "Other impeller collision"].copy()
print(f"  Leading-edge subset: {len(df_leading)} / {len(df)} passages")

leading_rows = []
for tx, grp in df_leading.groupby("treatment"):
    n = len(grp)
    k = int(grp["y_pred"].sum())
    p, lo, hi = wilson_ci(k, n)
    leading_rows.append({
        "method": "MiniRocket_leading", "treatment": tx,
        "n_fish": n, "n_pred_strike": k,
        "strike_prob": p, "ci_lower": lo, "ci_upper": hi,
    })
# Add "All data" aggregate for MiniRocket_leading
n_all_l = len(df_leading)
k_all_l = int(df_leading["y_pred"].sum())
p_all_l, lo_all_l, hi_all_l = wilson_ci(k_all_l, n_all_l)
leading_rows.append({
    "method": "MiniRocket_leading", "treatment": "All data",
    "n_fish": n_all_l, "n_pred_strike": k_all_l,
    "strike_prob": p_all_l, "ci_lower": lo_all_l, "ci_upper": hi_all_l,
})
leading_df = pd.DataFrame(leading_rows)
leading_df.to_csv(f"{OUTDIR}/leading_strike_probability.csv", index=False)

# Ground-truth strike counts per treatment for leading-edge subset (y_true, not y_pred)
leading_util = {
    tx: {"n": len(grp), "k": int(grp["y_true"].sum())}
    for tx, grp in df_leading.groupby("treatment")
}
leading_util["All data"] = {"n": n_all_l, "k": int(df_leading["y_true"].sum())}

# Threshold strike probabilities on the leading-edge subset (for panel B)
prob_leading_rows = []
for thr in THRESHOLDS:
    col = f"pred_{thr}g"
    # All data
    n_all = len(df_leading)
    k_all = int(df_leading[col].sum())
    p_all, lo_all, hi_all = wilson_ci(k_all, n_all)
    prob_leading_rows.append({
        "method": f"Threshold_{thr}g", "treatment": "All data",
        "n_fish": n_all, "strike_prob": p_all, "ci_lower": lo_all, "ci_upper": hi_all,
    })
    for tx, grp in df_leading.groupby("treatment"):
        n = len(grp); k = int(grp[col].sum())
        p, lo, hi = wilson_ci(k, n)
        prob_leading_rows.append({
            "method": f"Threshold_{thr}g", "treatment": tx,
            "n_fish": n, "strike_prob": p, "ci_lower": lo, "ci_upper": hi,
        })
prob_leading_df = pd.DataFrame(prob_leading_rows)


# ── Classification metrics ────────────────────────────────────────────────────
def classification_metrics(y_true, y_pred, label):
    tn  = ((y_true == 0) & (y_pred == 0)).sum()
    fp  = ((y_true == 0) & (y_pred == 1)).sum()
    fn  = ((y_true == 1) & (y_pred == 0)).sum()
    tp  = ((y_true == 1) & (y_pred == 1)).sum()
    n   = len(y_true)
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr         = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    precision   = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else np.nan)
    accuracy    = (tp + tn) / n
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc     = float(mcc_num / mcc_den) if mcc_den > 0 else 0.0
    return {
        "method": label, "n": n,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FPR": round(fpr, 4), "FNR": round(fnr, 4),
        "precision":   round(precision, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "F1":          round(f1, 4),
        "accuracy":    round(accuracy, 4),
        "MCC":         round(mcc, 4),
    }


methods = {
    "MiniRocket":     df["y_pred"].values,
    "Threshold_95g":  df["pred_95g"].values,
    "Threshold_200g": df["pred_200g"].values,
    "Threshold_400g": df["pred_400g"].values,
}

scores_map = {
    "MiniRocket":     df["probability"].values,
    "Threshold_95g":  df["max_acc_g"].values,
    "Threshold_200g": df["max_acc_g"].values,
    "Threshold_400g": df["max_acc_g"].values,
}

metrics_rows = [classification_metrics(y_true, preds, lbl)
                for lbl, preds in methods.items()]

for row in metrics_rows:
    row["AUC"] = round(roc_auc_score(y_true, scores_map[row["method"]]), 4)

mr_pred    = methods["MiniRocket"]
mr_correct = (mr_pred == y_true)

for row in metrics_rows:
    lbl = row["method"]
    if lbl == "MiniRocket":
        row["mcnemar_p"] = np.nan
        continue
    thr_correct = (methods[lbl] == y_true)
    b = int((mr_correct & ~thr_correct).sum())
    c = int((~mr_correct & thr_correct).sum())
    if (b + c) > 0:
        stat  = (abs(b - c) - 1) ** 2 / (b + c)
        p_val = float(chi2_dist.sf(stat, df=1))
    else:
        p_val = 1.0
    row["mcnemar_p"] = round(p_val, 4)

mr_score = scores_map["MiniRocket"]

for row in metrics_rows:
    lbl = row["method"]
    if lbl == "MiniRocket":
        row["delong_z"] = np.nan
        row["delong_p"] = np.nan
        continue
    z, p = delong_test(y_true, mr_score, scores_map[lbl])
    row["delong_z"] = round(z, 4)
    row["delong_p"] = round(p, 4)

metrics_df = pd.DataFrame(metrics_rows)
print(metrics_df[["method", "TP", "TN", "FP", "FN",
                   "FPR", "FNR", "precision", "sensitivity",
                   "specificity", "F1", "accuracy", "MCC",
                   "AUC", "mcnemar_p", "delong_z", "delong_p"]].to_string(index=False))
metrics_df.to_csv(f"{OUTDIR}/threshold_comparison_metrics.csv", index=False)


rows_by_type = []
for lbl, preds in methods.items():
    for stype in ["no_contact", "leading_indirect", "leading_direct",
                  "other_impeller_hub", "other_impeller_surface"]:
        mask = df["strike_type"] == stype
        if mask.sum() == 0:
            continue
        m = classification_metrics(y_true[mask], preds[mask], lbl)
        m["strike_type"] = stype
        m["n_files"]     = int(mask.sum())
        rows_by_type.append(m)

by_type_df = pd.DataFrame(rows_by_type)
print(by_type_df[["method", "strike_type", "n_files",
                   "FPR", "FNR", "accuracy"]].to_string(index=False))
by_type_df.to_csv(f"{OUTDIR}/threshold_by_strike_type.csv", index=False)


# ── Strike probability by treatment ──────────────────────────────────────────
prob_rows = []
for lbl, preds in methods.items():
    df_tmp = df.copy()
    df_tmp["_pred"] = preds
    # All data row
    n_all = len(df_tmp)
    k_all = int(preds.sum())
    p_all, lo_all, hi_all = wilson_ci(k_all, n_all)
    prob_rows.append({
        "method": lbl, "treatment": "All data",
        "n_fish": n_all, "n_true_strike": int(df["y_true"].sum()),
        "n_pred_strike": k_all, "strike_prob": p_all,
        "ci_lower": lo_all, "ci_upper": hi_all,
    })
    # Per treatment
    for tx, grp in df_tmp.groupby("treatment"):
        n = len(grp)
        k = grp["_pred"].sum()
        k_true = grp["y_true"].sum()
        p, lo, hi = wilson_ci(k, n)
        prob_rows.append({
            "method": lbl, "treatment": tx,
            "n_fish": n, "n_true_strike": int(k_true),
            "n_pred_strike": int(k), "strike_prob": p,
            "ci_lower": lo, "ci_upper": hi,
        })

# ── Colours, hatches, labels ─────────────────────────────────────────────────
METHOD_COLORS = {
    "total_obs":          "#bdbdbd",
    "total_util":         "#636363",
    "total_util_leading": "#636363",
    "MiniRocket":         "#1f78b4",
    "MiniRocket_leading": "#1f78b4",
    "esch_2014":          "#a6cee3",
    "cen_2025":           "#33a02c",
    "Threshold_400g":     "#b2df8a",
    "Threshold_200g":     "#fdbf6f",
    "Threshold_95g":      "#fb9a99",
}

METHOD_HATCHES = {
    "total_util_leading": "///",
    "MiniRocket_leading": "///",
}

METHOD_LABELS = {
    "total_obs":          "Total video observations",
    "total_util":         "Utilised video observations",
    "total_util_leading": "Utilised video (leading edge)",
    "MiniRocket":         "miniRocket (OOF CV)",
    "MiniRocket_leading": "miniRocket – leading edge",
    "esch_2014":          "van Esch & Spierts (2014)",
    "cen_2025":           "EVS-EN 18110 (2025)",
    "Threshold_400g":     "Acceleration θ ≥ 400g",
    "Threshold_200g":     "Acceleration θ ≥ 200g",
    "Threshold_95g":      "Acceleration θ ≥ 95g",
}

dark_methods = {"total_util", "total_util_leading", "MiniRocket", "MiniRocket_leading"}


def get_data(lbl, tx, metric, excl=False):
    """
    Universal data getter. metric = 'collision' | 'mortality'.
    excl=True  → use EXCL_REF (leading-edge subset).
    excl=False → use FULL_REF.
    Threshold methods fall back to prob_df / prob_leading_df (collision only).
    """
    ref_dict = EXCL_REF if excl else FULL_REF
    thr_set  = {"Threshold_400g", "Threshold_200g", "Threshold_95g"}

    if lbl in thr_set:
        if metric == "mortality":
            return np.nan, np.nan, np.nan, 0
        pool = prob_leading_df if excl else prob_df
        sub  = pool[(pool["method"] == lbl) & (pool["treatment"] == tx)]
        if len(sub) == 0:
            return np.nan, np.nan, np.nan, 0
        r = sub.iloc[0]
        return r["strike_prob"], r["ci_lower"], r["ci_upper"], int(r["n_fish"])

    return _ref_lookup(ref_dict, lbl, tx, metric)


def plot_bar(ax, x_pos, lbl, tx, bar_w, legend_entries, get_fn, error_bars=True, force_hatch=None):
    """
    Plot one bar. get_fn(lbl, tx) → (p, lo, hi, n).
    legend_entries is an ordered list of (lbl_key, handle) tuples.
    No text annotations inside bars.
    """
    p_hat, ci_lo, ci_hi, _ = get_fn(lbl, tx)
    if np.isnan(p_hat):
        return
    col   = METHOD_COLORS[lbl]
    hatch = force_hatch or METHOD_HATCHES.get(lbl, None)
    bars  = ax.bar(x_pos, p_hat, width=bar_w,
                   color=col, edgecolor="black", linewidth=0.6,
                   hatch=hatch, zorder=3)
    if not any(k == lbl for k, _ in legend_entries):
        legend_entries.append((lbl, bars[0]))
    if error_bars and not (np.isnan(ci_lo) or np.isnan(ci_hi)):
        ax.errorbar(x_pos, p_hat,
                    yerr=[[p_hat - ci_lo], [ci_hi - p_hat]],
                    fmt="none", color="#333333", capsize=2.5, lw=1.0,
                    capthick=1.0, zorder=4)


def style_axes(ax, x_lo, x_hi, metric="collision", show_ylabel=True, show_xlabel=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.spines["left"].set_bounds(0, 1.0)
    ax.spines["bottom"].set_bounds(x_lo, x_hi)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1.0)
    if show_ylabel:
        ylabel = ("Collision Probability (95% CI)" if metric == "collision"
                  else "Mortality Probability (95% CI)")
        ax.set_ylabel(ylabel, fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Treatment", fontsize=9)
    else:
        ax.set_xlabel("")


def apply_legend(ax, legend_entries, method_order, **kwargs):
    """Build legend from explicitly tracked (lbl_key, handle) list, ordered by method_order."""
    entry_dict = {lbl: h for lbl, h in legend_entries}
    handles, labels = [], []
    for lbl in method_order:
        if lbl in entry_dict:
            handles.append(entry_dict[lbl])
            labels.append(METHOD_LABELS[lbl])
    if handles:
        ax.legend(handles, labels, **kwargs)



prob_df = pd.DataFrame(prob_rows)

# ── Complete strike/mortality probability table ───────────────────────────────
TX3 = ["500 (100%)", "400 (100%)", "400 (70%)"]

def fmt_pct(p):
    return f"{p*100:.2f}%" if not (isinstance(p, float) and np.isnan(p)) else ""

def fmt_ci(lo, hi):
    if any(isinstance(v, float) and np.isnan(v) for v in [lo, hi]):
        return ""
    return f"[{lo*100:.2f} – {hi*100:.2f}%]"

out_rows = []

def add_out(panel, method, tx, metric, excl):
    p, lo, hi, n = get_data(method, tx, metric, excl=excl)
    if np.isnan(p):
        return
    out_rows.append({
        "panel":     panel,
        "metric":    metric,
        "method":    method,
        "treatment": tx,
        "n":         n,
        "estimate":  fmt_pct(p),
        "ci_95":     fmt_ci(lo, hi),
        "prob_raw":  round(p, 4),
        "ci_lo_raw": round(lo, 4) if not np.isnan(lo) else np.nan,
        "ci_hi_raw": round(hi, 4) if not np.isnan(hi) else np.nan,
    })

# Panel A rows (full dataset, collision): 8 methods
pa_collision = ["total_obs", "total_util", "MiniRocket",
                "esch_2014", "cen_2025",
                "Threshold_400g", "Threshold_200g", "Threshold_95g"]
# Panel B rows (full dataset, mortality): 4 methods
pb_mortality = ["total_util", "MiniRocket", "esch_2014", "cen_2025"]
# Panel C rows (excl, collision): 7 methods
pc_collision = ["total_util_leading", "MiniRocket_leading",
                "esch_2014", "cen_2025",
                "Threshold_400g", "Threshold_200g", "Threshold_95g"]
# Panel D rows (excl, mortality): 4 methods
pd_mortality = ["total_util_leading", "MiniRocket_leading", "esch_2014", "cen_2025"]

for tx in TX3:
    for m in pa_collision: add_out("A (collision, full)",    m, tx, "collision", excl=False)
    for m in pb_mortality:  add_out("B (mortality, full)",   m, tx, "mortality", excl=False)
    for m in pc_collision:  add_out("C (collision, excl)",   m, tx, "collision", excl=True)
    for m in pd_mortality:  add_out("D (mortality, excl)",   m, tx, "mortality", excl=True)

out_df = pd.DataFrame(out_rows)
print(out_df[["panel", "metric", "method", "treatment", "n", "estimate", "ci_95"]].to_string(index=False))
out_df.to_csv(f"{OUTDIR}/strike_probability_comparison.csv", index=False)

# ── Load MiniRocket OOF metrics ───────────────────────────────────────────────
try:
    with open(METRICS) as f:
        mr_metrics = json.load(f)
    oof     = mr_metrics["out_of_fold_performance"]
    mr_acc  = oof["overall_accuracy"]
    mr_sens = oof["sensitivity"]
    mr_spec = oof["specificity"]
    mr_auc  = oof["roc_auc"]
    mr_mcc  = oof.get("mcc", None)
    mr_info = True
except Exception as e:
    print(f"  WARNING: could not load metrics: {e}")
    mr_info = False


# ── Standalone FPR/FNR figure ─────────────────────────────────────────────────
method_labels_errs = ["MiniRocket", "Threshold_400g", "Threshold_200g", "Threshold_95g"]
cm_conv = 1 / 2.54
fig, ax = plt.subplots(figsize=(9 * cm_conv, 5.5 * cm_conv), dpi=300)
error_types     = ["FPR", "FNR"]
error_labels    = ["FPR (%)", "FNR (%)"]
n_methods       = len(method_labels_errs)
bar_w_err       = 1.0 / n_methods
group_positions = np.arange(len(error_types))

for i, lbl in enumerate(method_labels_errs):
    col     = METHOD_COLORS[lbl]
    offsets = group_positions + (i - n_methods / 2 + 0.5) * bar_w_err
    vals = [100 * metrics_df.loc[metrics_df["method"] == lbl, et].values[0]
            for et in error_types]
    bars = ax.bar(offsets, vals, width=bar_w_err,
                  color=col, edgecolor="black", label=METHOD_LABELS[lbl])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(group_positions)
ax.set_xticklabels(error_labels, fontsize=11)
ax.set_ylabel("Error rate (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_position(("outward", 5))
ax.spines["bottom"].set_position(("outward", 5))
ax.spines["left"].set_bounds(0, 100)
ax.spines["bottom"].set_bounds(0, 1.0)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#dddddd", zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fpr_fnr_comparison.svg", dpi=300, transparent=True)
plt.close()

# ── Standalone accuracy-by-strike-type figure ─────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
stype_order  = ["no_contact", "leading_indirect", "leading_direct",
                "other_impeller_hub", "other_impeller_surface"]
stype_labels = ["No Contact", "Leading Indirect", "Leading Direct",
                "Other: Hub", "Other: Surface"]
n_types  = len(stype_order)
n_m_fig2 = len(method_labels_errs)
bar_w_st = 0.7 / n_m_fig2

for i, lbl in enumerate(method_labels_errs):
    offsets = np.arange(n_types) + (i - n_m_fig2 / 2 + 0.5) * bar_w_st
    accs = []
    for stype in stype_order:
        sub = by_type_df[(by_type_df["method"] == lbl) & (by_type_df["strike_type"] == stype)]
        accs.append(sub["accuracy"].values[0] if len(sub) > 0 else 0)
    ax.bar(offsets, accs, width=bar_w_st,
           color=METHOD_COLORS[lbl], edgecolor="black", label=METHOD_LABELS[lbl])

ax.set_xticks(np.arange(n_types))
ax.set_xticklabels(stype_labels, fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_ylim(0, 1.12)
ax.axhline(1.0, color="gray", linestyle="--", lw=1)
ax.set_title("Classification Accuracy by Event Type", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/accuracy_by_strike_type.png", dpi=300, bbox_inches="tight")
plt.close()


# ── Figure geometry (shared) ─────────────────────────────────────────────────
cm    = 1 / 2.54
bar_w = 0.075
gap   = 0.028

g1, g2, g3, g4 = 2*bar_w, bar_w, 2*bar_w, 3*bar_w
total_w = g1 + g2 + g3 + g4 + 3*gap
half    = total_w / 2
s1 = -half
s2 = s1 + g1 + gap
s3 = s2 + g2 + gap
s4 = s3 + g3 + gap

OFF = {
    "total_obs":          s1 + 0.5*bar_w,
    "total_util":         s1 + 1.5*bar_w,
    "total_util_leading": s1 + 1.5*bar_w,
    "MiniRocket":         s2 + 0.5*bar_w,
    "MiniRocket_leading": s2 + 0.5*bar_w,
    "esch_2014":          s3 + 0.5*bar_w,
    "cen_2025":           s3 + 1.5*bar_w,
    "Threshold_400g":     s4 + 0.5*bar_w,
    "Threshold_200g":     s4 + 1.5*bar_w,
    "Threshold_95g":      s4 + 2.5*bar_w,
}

# Mortality panels: 4 bars [util | MR | esch | cen] centred
_m4 = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w
OFF_MORT = dict(zip(["total_util", "MiniRocket", "esch_2014", "cen_2025"], _m4))
OFF_MORT_EXCL = dict(zip(
    ["total_util_leading", "MiniRocket_leading", "esch_2014", "cen_2025"], _m4))

sp      = total_w * 1.22           # collision panels (8 bars)
sp_mort = (4 * bar_w) * 1.35      # mortality panels (4 bars, tighter)
TX3     = ["500 (100%)", "400 (100%)", "400 (70%)"]
tx_x      = {tx: i * sp      for i, tx in enumerate(TX3)}
tx_x_mort = {tx: i * sp_mort for i, tx in enumerate(TX3)}
thr_set = {"Threshold_400g", "Threshold_200g", "Threshold_95g"}

def render_panel(ax, methods, off_dict, tx_list, tx_positions,
                 metric, excl, legend_store, skip_thr_txs=None, error_bars=True, force_hatch=None):
    for tx in tx_list:
        xc = tx_positions[tx]
        for lbl in methods:
            if skip_thr_txs and tx in skip_thr_txs and lbl in thr_set:
                continue
            off = off_dict.get(lbl)
            if off is None:
                continue
            get_fn = lambda l, t, m=metric, e=excl: get_data(l, t, m, excl=e)
            plot_bar(ax, xc + off, lbl, tx, bar_w, legend_store, get_fn,
                     error_bars=error_bars, force_hatch=force_hatch)


# ── 2×2 main figure ──────────────────────────────────────────────────────────
#  a = collision full   b = mortality full
#  c = collision excl   d = mortality excl

pa_methods   = ["total_obs", "total_util", "MiniRocket",
                "esch_2014", "cen_2025",
                "Threshold_400g", "Threshold_200g", "Threshold_95g"]
pb_methods   = ["total_util", "MiniRocket", "esch_2014", "cen_2025"]
pc_methods   = ["total_util_leading", "MiniRocket_leading",
                "esch_2014", "cen_2025",
                "Threshold_400g", "Threshold_200g", "Threshold_95g"]
pd_methods   = ["total_util_leading", "MiniRocket_leading", "esch_2014", "cen_2025"]

from matplotlib.gridspec import GridSpec as _GS
fig = plt.figure(figsize=(24*cm, 22*cm))
gs  = _GS(3, 2, figure=fig,
          width_ratios=[2, 1],
          height_ratios=[1, 1, 1],
          hspace=0.45, wspace=0.35)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
ax_e = fig.add_subplot(gs[2, 1])   # panel e: scatter, same column/size as b/d

leg_a, leg_b, leg_c, leg_d = [], [], [], []

render_panel(ax_a, pa_methods, OFF,           TX3, tx_x,      "collision", False, leg_a)
render_panel(ax_b, pb_methods, OFF_MORT,      TX3, tx_x_mort, "mortality", False, leg_b, error_bars=False)
render_panel(ax_c, pc_methods, OFF,           TX3, tx_x,      "collision", True,  leg_c, force_hatch="///")
render_panel(ax_d, pd_methods, OFF_MORT_EXCL, TX3, tx_x_mort, "mortality", True,  leg_d, error_bars=False, force_hatch="///")

x_lo_col  = tx_x["500 (100%)"]
x_hi_col  = tx_x["400 (70%)"]
x_lo_mort = tx_x_mort["500 (100%)"]
x_hi_mort = tx_x_mort["400 (70%)"]

for ax, leg, met, title, show_xl, show_yl, x_lo, x_hi in [
    (ax_a, leg_a, "collision", "a", False, True,  x_lo_col,  x_hi_col),
    (ax_b, leg_b, "mortality", "b", False, True,  x_lo_mort, x_hi_mort),
    (ax_c, leg_c, "collision", "c", True,  True,  x_lo_col,  x_hi_col),
    (ax_d, leg_d, "mortality", "d", True,  True,  x_lo_mort, x_hi_mort),
]:
    style_axes(ax, x_lo, x_hi, metric=met, show_ylabel=show_yl, show_xlabel=show_xl)
    tx_pos = tx_x_mort if met == "mortality" else tx_x
    ax.set_xticks([tx_pos[tx] for tx in TX3])
    ax.set_xticklabels(TX3, fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")

# Legend in panel a only
apply_legend(
    ax_a, leg_a, pa_methods,
    fontsize=7.5, loc="upper right", ncol=2,
    frameon=True, edgecolor="#cccccc",
    bbox_to_anchor=(0.99, 0.99),
)

if mr_info:
    lines = ["miniRocket (OOF CV)",
             f"Accuracy :   {mr_acc:.3f}",
             f"Sensitivity: {mr_sens:.3f}",
             f"Specificity: {mr_spec:.3f}",
             f"AUC :        {mr_auc:.3f}"]
    if mr_mcc:
        lines.append(f"MCC :        {mr_mcc:.3f}")
    ax_a.text(0.02, 0.97, "\n".join(lines),
              transform=ax_a.transAxes, fontsize=7,
              va="top", ha="left",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"))

plt.savefig(f"{OUTDIR}/strike_probability_by_treatment.svg", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: strike_probability_by_treatment.svg  (2x2 panels)")


# ── Small "All data" summary figure ──────────────────────────────────────────
# 3 bars: total_obs | total_util | MiniRocket (collision only)
_ad_methods = ["total_obs", "total_util", "MiniRocket"]
_ad_off     = np.array([-1, 0, 1]) * bar_w

# Build "All data" values from total_obs ref (n=355,k=159) for total_obs
# and from prob_df for total_util and MiniRocket
_ad_data = {}
# total_obs all
n, k = 355, 159
p, lo, hi = wilson_ci(k, n)
_ad_data["total_obs"] = (p, lo, hi, n)
# total_util all
n, k = 258, 118
p, lo, hi = wilson_ci(k, n)
_ad_data["total_util"] = (p, lo, hi, n)
# MiniRocket all — from prob_df
sub = prob_df[(prob_df["method"] == "MiniRocket") & (prob_df["treatment"] == "All data")]
if len(sub):
    r = sub.iloc[0]
    _ad_data["MiniRocket"] = (r["strike_prob"], r["ci_lower"], r["ci_upper"], int(r["n_fish"]))

fig_ad, ax_ad = plt.subplots(figsize=(8*cm, 7*cm))
leg_ad = []
for i, lbl in enumerate(_ad_methods):
    if lbl not in _ad_data:
        continue
    p_hat, ci_lo, ci_hi, _ = _ad_data[lbl]
    col   = METHOD_COLORS[lbl]
    hatch = METHOD_HATCHES.get(lbl, None)
    bars  = ax_ad.bar(_ad_off[i], p_hat, width=bar_w,
                      color=col, edgecolor="black", linewidth=0.6,
                      hatch=hatch, zorder=3)
    leg_ad.append((lbl, bars[0]))
    if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
        ax_ad.errorbar(_ad_off[i], p_hat,
                       yerr=[[p_hat - ci_lo], [ci_hi - p_hat]],
                       fmt="none", color="#333333", capsize=2.5, lw=1.0,
                       capthick=1.0, zorder=4)

ax_ad.spines["top"].set_visible(False)
ax_ad.spines["right"].set_visible(False)
ax_ad.spines["left"].set_position(("outward", 5))
ax_ad.spines["bottom"].set_position(("outward", 5))
ax_ad.spines["left"].set_bounds(0, 1.0)
ax_ad.spines["bottom"].set_bounds(_ad_off[0], _ad_off[-1])
ax_ad.spines["left"].set_linewidth(1.0)
ax_ad.spines["bottom"].set_linewidth(1.0)
ax_ad.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#dddddd", zorder=0)
ax_ad.set_axisbelow(True)
ax_ad.set_yticks(np.arange(0, 1.1, 0.1))
ax_ad.set_ylim(0, 1.0)
ax_ad.set_ylabel("Collision Probability (95% CI)", fontsize=9)
ax_ad.set_xticks([])
apply_legend(ax_ad, leg_ad, _ad_methods,
             fontsize=8, loc="upper right", frameon=True, edgecolor="#cccccc")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/alldata_summary.svg", dpi=300)
plt.close()
print("  Saved: alldata_summary.svg")


# ── Panel e: Collision vs Mortality scatter (embedded in main figure) ─────────
_SCATTER = [
    ("500 (100%)",  [10,20,30,40,50,60,70,80,90,100],
                    [3.88,7.76,11.64,15.52,19.4,23.28,27.16,31.04,34.92,38.8]),
    ("400 (100%)",  [10,20,30,40,50,60,70,80,90,100],
                    [2.23,4.44,6.67,8.89,11.11,13.34,15.56,17.87,20.01,22.23]),
    ("400 (70%)",   [10,20,30,40,50,60,70,80,90,100],
                    [1.99,3.99,5.98,7.98,9.97,11.97,13.96,15.96,17.96,19.95]),
]
TX_COLORS = {
    "500 (100%)": "#e41a1c",
    "400 (100%)": "#377eb8",
    "400 (70%)":  "#4daf4a",
}

for tx, col_vals, mort_vals in _SCATTER:
    slope = mort_vals[-1] / col_vals[-1]   # linear: mort = slope * collision
    ax_e.plot(col_vals, mort_vals, color=TX_COLORS[tx],
              linewidth=1.5, label=f"{tx}  (slope = {slope:.2f})", zorder=3)

ax_e.spines["top"].set_visible(False)
ax_e.spines["right"].set_visible(False)
ax_e.spines["left"].set_position(("outward", 5))
ax_e.spines["bottom"].set_position(("outward", 5))
ax_e.spines["left"].set_linewidth(1.0)
ax_e.spines["bottom"].set_linewidth(1.0)
ax_e.set_xlabel("Collision Probability (%)", fontsize=9)
ax_e.set_ylabel("Mortality Probability (%)", fontsize=9)
ax_e.set_xlim(0, 100)
ax_e.set_ylim(0, 100)
ax_e.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#dddddd", zorder=0)
ax_e.set_axisbelow(True)
ax_e.legend(fontsize=7.5, frameon=True, edgecolor="#cccccc", loc="upper left")
ax_e.set_title("e", loc="left", fontsize=11, fontweight="bold")

# Save main figure (now includes panel e)
fig.savefig(f"{OUTDIR}/strike_probability_by_treatment.svg", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: strike_probability_by_treatment.svg  (panels a-e)")

print(f"\nAll outputs saved to: {OUTDIR}/")
