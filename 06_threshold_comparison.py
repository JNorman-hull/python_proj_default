import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os


RAW_DATA    = "python_data/labeled_data_with_types.csv"
CV_PREDS    = "python_results/binary/cv_predictions.csv"
METRICS     = "python_results/binary/performance_metrics.json"
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

df = cv.merge(acc_max, on="file", how="inner")
print(f"  Files matched after merge: {len(df)}")
if len(df) < len(cv):
    print(f"  WARNING: {len(cv) - len(df)} CV files not found in filtered raw data")

y_true = df["y_true"].values

for thr in THRESHOLDS:
    df[f"pred_{thr}g"] = (df["max_acc_g"] >= thr).astype(int)


def classification_metrics(y_true, y_pred, label):
    tn  = ((y_true == 0) & (y_pred == 0)).sum()
    fp  = ((y_true == 0) & (y_pred == 1)).sum()
    fn  = ((y_true == 1) & (y_pred == 0)).sum()
    tp  = ((y_true == 1) & (y_pred == 1)).sum()
    n   = len(y_true)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    accuracy  = (tp + tn) / n
    return {
        "method": label, "n": n,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FPR": round(fpr, 4), "FNR": round(fnr, 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "F1": round(f1, 4), "accuracy": round(accuracy, 4),
    }

methods = {
    "MiniRocket":     df["y_pred"].values,
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
            "method": lbl, "treatment": tx,
            "n_fish": n, "n_true_strike": int(k_true),
            "n_pred_strike": int(k), "strike_prob": p,
            "ci_lower": lo, "ci_upper": hi,
        })

prob_df = pd.DataFrame(prob_rows)
print(prob_df.to_string(index=False))
prob_df.to_csv(f"{OUTDIR}/strike_probability_comparison.csv", index=False)

method_labels = ["MiniRocket", "Threshold_400g", "Threshold_200g", "Threshold_95g"]

METHOD_COLORS = {
    "Observed2":      "#bdbdbd",  # light grey
    "Van Esch":       "#a6cee3",  # light blue
    "Observed":       "#636363",  # mid grey
    "MiniRocket":     "#1f78b4",  # blue
    "Threshold_400g": "#b2df8a",  # light green
    "Threshold_200g": "#fdbf6f",  # light amber
    "Threshold_95g":  "#fb9a99",  # light red
}

METHOD_LABELS = {
    "Observed2":      "Total Video observations",
    "Van Esch":       "van Esch & Spierts 2014*",
    "Observed":       "Utilised Video observations",
    "MiniRocket":     "miniRocket (OOF CV)",
    "Threshold_400g": "Acceleration θ ≥ 400g",
    "Threshold_200g": "Acceleration θ ≥ 200g",
    "Threshold_95g":  "Acceleration θ ≥ 95g",
}

# ── Fig 1: FPR / FNR ─────────────────────────────────────────────────────────
cm_conv = 1 / 2.54
fig, ax = plt.subplots(figsize=(14 * cm_conv, 8 * cm_conv), dpi=300)
error_types  = ["FPR", "FNR"]
error_labels = [
    "False Positive Rate (%)",
    "False Negative Rate (%))"
]
n_methods    = len(method_labels)
group_w      = 0.75
bar_w        = group_w / n_methods
group_positions = np.arange(len(error_types))

for i, lbl in enumerate(method_labels):
    col     = METHOD_COLORS[lbl]
    offsets = group_positions + (i - n_methods / 2 + 0.5) * bar_w
    vals = [100 * metrics_df.loc[metrics_df["method"] == lbl, et].values[0]
            for et in error_types]
    bars = ax.bar(offsets, vals, width=bar_w * 0.9,
                  color=col, edgecolor="black", label=METHOD_LABELS[lbl])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(group_positions)
ax.set_xticklabels(error_labels, fontsize=11)
ax.set_ylabel("Error rate (%)", fontsize=11)
ax.set_ylim(0, 100)
# ── Spines ────────────────────────────────────────────────────────────────────
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

ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fpr_fnr_comparison.svg", dpi=300)
plt.close()

# ── Fig 2: Accuracy by strike type ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
stype_order  = ["no_contact", "leading_indirect", "leading_direct",
                "other_impeller_hub", "other_impeller_surface"]
stype_labels = ["No Contact", "Leading Indirect", "Leading Direct",
                "Other: Hub", "Other: Surface"]
n_types = len(stype_order)
bar_w2  = 0.75 / n_methods

for i, lbl in enumerate(method_labels):
    offsets = np.arange(n_types) + (i - n_methods / 2 + 0.5) * bar_w2
    accs = []
    for stype in stype_order:
        sub = by_type_df[(by_type_df["method"] == lbl) & (by_type_df["strike_type"] == stype)]
        accs.append(sub["accuracy"].values[0] if len(sub) > 0 else 0)
    ax.bar(offsets, accs, width=bar_w2 * 0.9,
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

# ── Reference data ────────────────────────────────────────────────────────────
VAN_ESCH_P = {
    "500 (100%)": 0.35,
    "400 (100%)": 0.35,
    "400 (70%)":  0.50,
}
OBSERVED2 = {
    "400 (100%)": {"n": 130, "k": 51},
    "400 (70%)":  {"n": 117, "k": 57},
    "500 (100%)": {"n": 108, "k": 48},
}

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

# ── Fig 3: Strike probability by treatment ────────────────────────────────────

cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(24 * cm, 14 * cm))

all_methods_plot = [ "Van Esch","Observed2", "Observed", "MiniRocket",
                    "Threshold_400g", "Threshold_200g", "Threshold_95g"]
tx_order = ["500 (100%)", "400 (100%)", "400 (70%)"]

# Subgroup offsets: no gap within group, inter_gap between groups
bar_w         = 0.09
inter_gap     = 0.03
subgroup_sizes = [1, 1, 2, 3]  # [Observed2+VanEsch] [Observed+MiniRocket] [400+200+95]
total_w = sum(subgroup_sizes) * bar_w + (len(subgroup_sizes) - 1) * inter_gap

pos = -total_w / 2 + bar_w / 2
offsets_within = []
for sg_i, sg in enumerate(subgroup_sizes):
    for _ in range(sg):
        offsets_within.append(pos)
        pos += bar_w
    if sg_i < len(subgroup_sizes) - 1:
        pos += inter_gap

# dark methods get white text inside bars
dark_methods = {"Observed", "MiniRocket"}


for i, lbl in enumerate(all_methods_plot):
    col = METHOD_COLORS[lbl]
    for j, tx in enumerate(tx_order):
        x_pos = j + offsets_within[i]

        if lbl == "Van Esch":
            p_hat = VAN_ESCH_P[tx]
            n_ref = OBSERVED[tx]["n"]
            _, ci_lo, ci_hi = wilson_ci(round(p_hat * n_ref), n_ref)
        elif lbl == "Observed2":
            n_ref = OBSERVED2[tx]["n"]
            p_hat, ci_lo, ci_hi = wilson_ci(OBSERVED2[tx]["k"], n_ref)
        elif lbl == "Observed":
            n_ref = OBSERVED[tx]["n"]
            p_hat, ci_lo, ci_hi = wilson_ci(OBSERVED[tx]["k"], n_ref)
        else:
            sub = prob_df[(prob_df["method"] == lbl) & (prob_df["treatment"] == tx)]
            if len(sub) == 0:
                continue
            p_hat = sub["strike_prob"].values[0]
            ci_lo = sub["ci_lower"].values[0]
            ci_hi = sub["ci_upper"].values[0]
            n_ref = int(sub["n_fish"].values[0])

        ax.bar(x_pos, p_hat, width=bar_w,
               color=col, edgecolor="black", linewidth=0.6,
               label=METHOD_LABELS[lbl] if j == 0 else "_nolegend_")
        ax.errorbar(x_pos, p_hat,
                    yerr=[[p_hat - ci_lo], [ci_hi - p_hat]],
                    fmt="none", color="#333333", capsize=2.5, lw=1.0, capthick=1.0)

        txt_col = "white" if lbl in dark_methods else "black"

        # percentage label at y=0.08
        ax.text(x_pos, 0.08, f"{p_hat*100:.2f}%",
                ha="center", va="bottom", fontsize=6.5,
                rotation=90, color=txt_col, fontweight="bold")

        # n= label at y=0.30, italic
        ax.text(x_pos, 0.20, f"n={n_ref}",
                ha="center", va="bottom", fontsize=6,
                rotation=90, color=txt_col, style="italic")

# ── Spines ────────────────────────────────────────────────────────────────────
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_position(("outward", 5))
ax.spines["bottom"].set_position(("outward", 5))
ax.spines["left"].set_bounds(0, 1.0)
ax.spines["bottom"].set_bounds(0, 2.0)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

# ── Grid & axes ───────────────────────────────────────────────────────────────
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#dddddd", zorder=0)
ax.set_axisbelow(True)
ax.set_xticks(np.arange(len(tx_order)))
ax.set_xticklabels(tx_order, fontsize=11)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylim(0, 1.0)
ax.set_ylabel("Blade Strike Probability (95% CI)", fontsize=11)
ax.set_xlabel("Treatment", fontsize=11)

# ── Legend (inside upper right) ───────────────────────────────────────────────
handles, labels_leg = ax.get_legend_handles_labels()
handle_dict     = dict(zip(labels_leg, handles))
ordered_handles = [handle_dict[METHOD_LABELS[l]] for l in all_methods_plot
                   if METHOD_LABELS[l] in handle_dict]
ordered_labels  = [METHOD_LABELS[l] for l in all_methods_plot
                   if METHOD_LABELS[l] in handle_dict]
ax.legend(ordered_handles, ordered_labels,
          fontsize=8.5, loc="upper right", ncol=2,
          frameon=True, edgecolor="#cccccc",
          bbox_to_anchor=(0.99, 0.99))

# ── MiniRocket stats box (upper left) ─────────────────────────────────────────
if mr_info:
    lines = ["miniRocket (OOF CV)",
             f"Accuracy :   {mr_acc:.3f}",
             f"Sensitivity: {mr_sens:.3f}",
             f"Specificity: {mr_spec:.3f}",
             f"AUC :        {mr_auc:.3f}"]
    if mr_mcc:
        lines.append(f"MCC :        {mr_mcc:.3f}")
    ax.text(0.02, 0.97, "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#cccccc"))

plt.tight_layout()
plt.savefig(f"{OUTDIR}/strike_probability_by_treatment.svg", dpi=300)
plt.close()

print(f"\nAll outputs saved to: {OUTDIR}/")
