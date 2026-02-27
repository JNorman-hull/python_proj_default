import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


RAW_DATA    = "python_data/labeled_data_with_types.csv"
CV_PREDS    = "python_results/binary/cv_predictions.csv"
OUTDIR      = "python_results/threshold_comparison"
os.makedirs(OUTDIR, exist_ok=True)

THRESHOLDS  = [95, 200, 400]   # g


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return round(p, 4), round(max(0, centre - margin), 4), round(min(1, centre + margin), 4)


raw = pd.read_csv(RAW_DATA, low_memory=False)
raw = raw[raw["passage_type"] != "Other impeller collision"].copy()

# Max acceleration magnitude per file
acc_max = (
    raw.groupby("file")["higacc_mag_g"]
    .max()
    .reset_index()
    .rename(columns={"higacc_mag_g": "max_acc_g"})
)
print(f"  Unique files after filter: {len(acc_max)}")

cv = pd.read_csv(CV_PREDS)
print(f"  Files in CV predictions: {len(cv)}")

# Merge acceleration max onto CV predictions
df = cv.merge(acc_max, on="file", how="inner")
print(f"  Files matched after merge: {len(df)}")
if len(df) < len(cv):
    print(f"  WARNING: {len(cv) - len(df)} CV files not found in filtered raw data")

# Ensure y_true is binary (0 = no strike, 1 = strike)
y_true = df["y_true"].values

# ── 3. Apply deterministic thresholds ────────────────────────────────────────
for thr in THRESHOLDS:
    df[f"pred_{thr}g"] = (df["max_acc_g"] >= thr).astype(int)

# ── 4. Compute per-method metrics ────────────────────────────────────────────
def classification_metrics(y_true, y_pred, label):
    tn  = ((y_true == 0) & (y_pred == 0)).sum()
    fp  = ((y_true == 0) & (y_pred == 1)).sum()
    fn  = ((y_true == 1) & (y_pred == 0)).sum()
    tp  = ((y_true == 1) & (y_pred == 1)).sum()
    n   = len(y_true)

    fpr       = fp / (fp + tn) if (fp + tn) > 0 else np.nan   # false positive rate
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else np.nan   # false negative rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    accuracy  = (tp + tn) / n

    return {
        "method":    label,
        "n":         n,
        "TP":        int(tp), "TN": int(tn),
        "FP":        int(fp), "FN": int(fn),
        "FPR":       round(fpr,  4),
        "FNR":       round(fnr,  4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "F1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
    }

methods = {
    "MiniRocket":  df["y_pred"].values,
    "Threshold_95g":  df["pred_95g"].values,
    "Threshold_200g": df["pred_200g"].values,
    "Threshold_400g": df["pred_400g"].values,
}

metrics_rows = [classification_metrics(y_true, preds, lbl)
                for lbl, preds in methods.items()]
metrics_df = pd.DataFrame(metrics_rows)

print(metrics_df[["method", "TP", "TN", "FP", "FN",
                   "FPR", "FNR", "precision", "recall", "F1", "accuracy"]].to_string(index=False))
metrics_df.to_csv(f"{OUTDIR}/threshold_comparison_metrics.csv", index=False)

strike_types = df["strike_type"].unique()
rows_by_type = []

for lbl, preds in methods.items():
    for stype in ["no_contact", "leading_indirect", "leading_direct"]:
        mask = df["strike_type"] == stype
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = preds[mask]
        m  = classification_metrics(yt, yp, lbl)
        m["strike_type"] = stype
        m["n_files"]     = int(mask.sum())
        rows_by_type.append(m)

by_type_df = pd.DataFrame(rows_by_type)
print(by_type_df[["method", "strike_type", "n_files",
                   "FPR", "FNR", "accuracy"]].to_string(index=False))
by_type_df.to_csv(f"{OUTDIR}/threshold_by_strike_type.csv", index=False)

prob_rows = []
for lbl, preds in methods.items():
    df_tmp = df.copy()
    df_tmp["_pred"] = preds
    for tx, grp in df_tmp.groupby("treatment"):
        n      = len(grp)
        k      = grp["_pred"].sum()
        k_true = grp["y_true"].sum()
        p, lo, hi = wilson_ci(k, n)
        prob_rows.append({
            "method":        lbl,
            "treatment":     tx,
            "n_fish":        n,
            "n_true_strike": int(k_true),
            "n_pred_strike": int(k),
            "strike_prob":   p,
            "ci_lower":      lo,
            "ci_upper":      hi,
        })

prob_df = pd.DataFrame(prob_rows)
print(prob_df.to_string(index=False))
prob_df.to_csv(f"{OUTDIR}/strike_probability_comparison.csv", index=False)

method_labels = ["MiniRocket", "Threshold_400g", "Threshold_200g", "Threshold_95g"]
# Colours: MiniRocket, 95g, 200g, 400g, Van Esch, Observed
METHOD_COLORS = {
    "MiniRocket":     "#2c7bb6",
    "Threshold_95g":  "#d7191c",
    "Threshold_200g": "#fdae61",
    "Threshold_400g": "#1a9641",
    "Van Esch":       "#7b2d8b",
    "Observed":       "#636363",
}
# ── Fig 1: FPR / FNR grouped by error type ───────────────────────────────────
# One group per error type (FPR, FNR); one bar per method within group

cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(14 * cm, 8 * cm), dpi=300)

error_types  = ["FPR", "FNR"]
error_labels = [
    "False Positive Rate (%)\n(no-contact classified as strike)",
    "False Negative Rate (%)\n(strike classified as no-contact)"
]

n_methods = len(method_labels)
group_w   = 0.75
bar_w     = group_w / n_methods
group_positions = np.arange(len(error_types))

for i, lbl in enumerate(method_labels):
    col     = METHOD_COLORS[lbl]
    offsets = group_positions + (i - n_methods / 2 + 0.5) * bar_w

    # convert to %
    vals = [
        100 * metrics_df.loc[metrics_df["method"] == lbl, et].values[0]
        for et in error_types
    ]

    bars = ax.bar(
        offsets, vals,
        width=bar_w * 0.9,
        color=col,
        edgecolor="black",
        label=lbl
    )

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{v:.1f}%",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold"
        )

ax.set_xticks(group_positions)
ax.set_xticklabels(error_labels, fontsize=11)
ax.set_ylabel("Error rate (%)", fontsize=11)
ax.set_ylim(0, 40)
ax.set_title(
    "False Positive & False Negative Rates by Method",
    fontsize=11, fontweight="bold"
)

ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fpr_fnr_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# ── Fig 2: Accuracy by strike type ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
stype_order  = ["no_contact", "leading_indirect", "leading_direct"]
stype_labels = ["No Contact", "Leading Indirect", "Leading Direct"]
n_types  = len(stype_order)
bar_w2   = 0.75 / n_methods

for i, lbl in enumerate(method_labels):
    offsets = np.arange(n_types) + (i - n_methods / 2 + 0.5) * bar_w2
    accs = []
    for stype in stype_order:
        sub = by_type_df[(by_type_df["method"] == lbl) & (by_type_df["strike_type"] == stype)]
        accs.append(sub["accuracy"].values[0] if len(sub) > 0 else 0)
    bars = ax.bar(offsets, accs, width=bar_w2 * 0.9,
                  color=METHOD_COLORS[lbl], edgecolor="black", label=lbl)

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
# Reference data: Van Esch model predictions and observed proportions
# CI for Van Esch uses Wilson interval applied to (p_van_esch * n_observed) / n_observed
# i.e. treating the model estimate as if observed from our sample size
VAN_ESCH_P = {
    "500 (100%)": 0.35,
    "400 (100%)": 0.35,
    "400 (70%)":  0.50,
}
# does not include hub and surface strikes
OBSERVED2 = {
    "400 (100%)": {"n": 130, "k": 47},
    "400 (70%)":  {"n": 117, "k": 51},
    "500 (100%)": {"n": 108, "k": 39},
}
OBSERVED2["Overall"] = {
    "n": sum(OBSERVED2[t]["n"] for t in ["500 (100%)", "400 (100%)", "400 (70%)"]),
    "k": sum(OBSERVED2[t]["k"] for t in ["500 (100%)", "400 (100%)", "400 (70%)"]),
}

# Derive observed counts directly from raw data
obs_file = (
    raw.groupby("file")[["treatment", "passage_type"]]
    .first()
    .reset_index()
)
obs_file["strike"] = (obs_file["passage_type"] != "No contact").astype(int)
OBSERVED = {
    tx: {"n": len(grp), "k": int(grp["strike"].sum())}
    for tx, grp in obs_file.groupby("treatment")
    if tx in VAN_ESCH_P
}

# Add Overall row to prob_df for each method
overall_rows = []
for lbl, preds in methods.items():
    df_tmp = df.copy()
    df_tmp["_pred"] = preds
    n      = len(df_tmp)
    k      = df_tmp["_pred"].sum()
    k_true = df_tmp["y_true"].sum()
    p, lo, hi = wilson_ci(k, n)
    overall_rows.append({
        "method": lbl, "treatment": "Overall",
        "n_fish": n, "n_true_strike": int(k_true),
        "n_pred_strike": int(k), "strike_prob": p,
        "ci_lower": lo, "ci_upper": hi,
    })
prob_df = pd.concat([prob_df, pd.DataFrame(overall_rows)], ignore_index=True)

# Observed overall — pool across the 3 ML-comparable treatments
obs_tx = ["500 (100%)", "400 (100%)", "400 (70%)"]
OBSERVED["Overall"] = {
    "n": sum(OBSERVED[t]["n"] for t in obs_tx),
    "k": sum(OBSERVED[t]["k"] for t in obs_tx),
}

METHOD_COLORS["Observed2"] = "#2ca25f"

tx_order = ["Overall", "500 (100%)", "400 (100%)", "400 (70%)"]
all_methods_plot = ["Van Esch", "Observed", "Observed2"] + method_labels
n_methods_plot   = len(all_methods_plot)
bar_w3 = 0.82 / n_methods_plot

fig, ax = plt.subplots(figsize=(15, 6))

for i, lbl in enumerate(all_methods_plot):
    col     = METHOD_COLORS[lbl]
    offsets = np.arange(len(tx_order)) + (i - n_methods_plot / 2 + 0.5) * bar_w3

    for j, tx in enumerate(tx_order):
        if lbl == "Van Esch" and tx == "Overall":
            continue
        if lbl in ["Van Esch", "Observed", "Observed2"]:
            if lbl == "Van Esch":
                p_hat = VAN_ESCH_P[tx]
                n_ref = OBSERVED[tx]["n"]
                k_ref = round(p_hat * n_ref)
                _, ci_lo, ci_hi = wilson_ci(k_ref, n_ref)
            elif lbl == "Observed2":
                n_ref = OBSERVED2[tx]["n"]
                k_ref = OBSERVED2[tx]["k"]
                p_hat, ci_lo, ci_hi = wilson_ci(k_ref, n_ref)
            else:  # Observed
                n_ref = OBSERVED[tx]["n"]
                k_ref = OBSERVED[tx]["k"]
                p_hat, ci_lo, ci_hi = wilson_ci(k_ref, n_ref)
            err_lo = p_hat - ci_lo
            err_hi = ci_hi - p_hat
        else:
            sub = prob_df[(prob_df["method"] == lbl) & (prob_df["treatment"] == tx)]
            if len(sub) == 0:
                continue
            p_hat  = sub["strike_prob"].values[0]
            err_lo = p_hat - sub["ci_lower"].values[0]
            err_hi = sub["ci_upper"].values[0] - p_hat

        bar = ax.bar(offsets[j], p_hat, width=bar_w3 * 0.88,
                     color=col, edgecolor="black",
                     label=lbl if j == 0 else "_nolegend_",
                     alpha=0.88)
        ax.errorbar(offsets[j], p_hat,
                    yerr=[[err_lo], [err_hi]],
                    fmt="none", color="black", capsize=3, lw=1.2)
        ax.text(offsets[j], p_hat + err_hi + 0.015,
                f"{p_hat*100:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", rotation=90)

ax.set_xticks(np.arange(len(tx_order)))
ax.set_xticklabels(tx_order, fontsize=11)
ax.set_ylabel("Blade Strike Probability (95% Wilson CI)", fontsize=12)
ax.set_ylim(0, 1)
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles, labels_leg, fontsize=11, loc="upper left",
          ncol=2, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/strike_probability_by_treatment.png", dpi=300, bbox_inches="tight")
plt.close()


print(f"\nAll outputs saved to: {OUTDIR}/")
