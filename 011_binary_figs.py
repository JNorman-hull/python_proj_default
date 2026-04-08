import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, confusion_matrix

BINARY_DIR = "python_results/binary"
THR_DIR    = "python_results/threshold_comparison"

# ── Load data ─────────────────────────────────────────────────────────────────
thr_df     = pd.read_csv(f"{THR_DIR}/threshold_predictions.csv")
metrics_df = pd.read_csv(f"{THR_DIR}/threshold_comparison_metrics.csv")

y_true = thr_df["y_true"].values

METHODS = {
    "MiniRocket":     {"pred": thr_df["y_pred"].values,    "score": thr_df["probability"].values},
    "Threshold_400g": {"pred": thr_df["pred_400g"].values, "score": thr_df["max_acc_g"].values},
    "Threshold_200g": {"pred": thr_df["pred_200g"].values, "score": thr_df["max_acc_g"].values},
    "Threshold_95g":  {"pred": thr_df["pred_95g"].values,  "score": thr_df["max_acc_g"].values},
}

METHOD_LABELS = {
    "MiniRocket":     "miniRocket (OOF CV)",
    "Threshold_400g": "Acceleration θ ≥ 400g",
    "Threshold_200g": "Acceleration θ ≥ 200g",
    "Threshold_95g":  "Acceleration θ ≥ 95g",
}

METHOD_COLORS = {
    "MiniRocket":     "#1f78b4",
    "Threshold_400g": "#b2df8a",
    "Threshold_200g": "#fdbf6f",
    "Threshold_95g":  "#fb9a99",
}

CM_DARK = {
    "MiniRocket":     "#1f78b4",
    "Threshold_400g": "#33a02c",
    "Threshold_200g": "#ff7f00",
    "Threshold_95g":  "#e31a1c",
}

CM_CMAPS = {
    lbl: LinearSegmentedColormap.from_list(lbl, ["#ffffff", CM_DARK[lbl]])
    for lbl in METHOD_COLORS
}

THR_GVALS = {
    "Threshold_400g": 400,
    "Threshold_200g": 200,
    "Threshold_95g":  95,
}

method_order = ["MiniRocket", "Threshold_400g", "Threshold_200g", "Threshold_95g"]
cm_conv      = 1 / 2.54


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — 2×2 Confusion Matrices  (12 × 8 cm)
# Layout: [MiniRocket | Threshold_400g]
#         [Threshold_200g | Threshold_95g]
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib.gridspec as gridspec
fig    = plt.figure(figsize=(22 * cm_conv, 8 * cm_conv), dpi=300)
outer  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.20)
cm_gs  = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0],
                                          hspace=0.3, wspace=0.1)
axes   = np.array([[fig.add_subplot(cm_gs[r, c]) for c in range(2)] for r in range(2)])
grid_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]

for (row_i, col_i), lbl in zip(grid_pos, method_order):
    ax     = axes[row_i, col_i]
    pred   = METHODS[lbl]["pred"]
    cm_mat = confusion_matrix(y_true, pred)
    cm_pct = cm_mat.astype(float) / cm_mat.sum(axis=1)[:, np.newaxis] * 100
    annot  = np.array([[f'{cm_mat[i,j]}\n({cm_pct[i,j]:.1f}%)'
                        for j in range(2)] for i in range(2)])

    sns.heatmap(
        cm_mat, annot=annot, fmt="",
        cmap=CM_CMAPS[lbl],
        xticklabels=["No Strike", "Strike"],
        yticklabels=["No Strike", "Strike"],
        annot_kws={"size": 7},
        ax=ax,
        linewidths=0.5,
        linecolor="black",
        cbar=True,
        cbar_kws={"shrink": 0.8, "pad": 0.02},
    )

    # Black border on colourbar + ticks every 20
    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1)
    cbar_max = int(cm_mat.max())
    cbar.set_ticks(range(0, cbar_max + 1, 20))
    cbar.ax.tick_params(labelsize=6)

    # Border on heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1)

    ax.set_title(METHOD_LABELS[lbl], fontsize=7, fontweight="bold", pad=4,
                 color=CM_DARK[lbl])
    ax.set_ylabel("True Label" if col_i == 0 else "", fontsize=7)
    ax.set_xlabel("Predicted Label" if row_i == 1 else "", fontsize=7)
    ax.tick_params(labelsize=6)

    mcc_val = metrics_df.loc[metrics_df["method"] == lbl, "MCC"].values[0]
    ax.text(0.98, 0.02, f"MCC = {mcc_val:.3f}",
            transform=ax.transAxes, fontsize=5.5,
            va="bottom", ha="right",
           )

plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)


# ══════════════════════════════════════════════════════════════════════════════
# Right panel — Single ROC plot
# ══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(outer[1])

# ── MiniRocket ROC curve ──────────────────────────────────────────────────────
mr_score = METHODS["MiniRocket"]["score"]
mr_fpr, mr_tpr, _ = roc_curve(y_true, mr_score)
mr_auc_val = metrics_df.loc[metrics_df["method"] == "MiniRocket", "AUC"].values[0]

ax.plot(mr_fpr, mr_tpr,
        color=METHOD_COLORS["MiniRocket"], lw=1,
        label=f"miniRocket",
        zorder=2)

# MiniRocket optimal operating point (Youden's J)
opt_idx = np.argmax(mr_tpr - mr_fpr)
ax.scatter(mr_fpr[opt_idx], mr_tpr[opt_idx],
           color=METHOD_COLORS["MiniRocket"], s=40, zorder=5,
           edgecolors="black", lw=0.8)

# ── Shared acceleration threshold ROC curve ───────────────────────────────────
thr_score = METHODS["Threshold_400g"]["score"]
thr_fpr, thr_tpr, _ = roc_curve(y_true, thr_score)
thr_auc_val = metrics_df.loc[metrics_df["method"] == "Threshold_400g", "AUC"].values[0]

ax.plot(thr_fpr, thr_tpr,
        color="black", lw=1, linestyle="-",
        label=f"Acceleration θ",
        zorder=2)

# Three operating points — exact FPR/TPR at each g-value
neg_scores = thr_score[y_true == 0]
pos_scores = thr_score[y_true == 1]

for thr_lbl in ["Threshold_400g", "Threshold_200g", "Threshold_95g"]:
    thr_g  = THR_GVALS[thr_lbl]
    fpr_pt = np.mean(neg_scores >= thr_g)
    tpr_pt = np.mean(pos_scores >= thr_g)
    ax.scatter(fpr_pt, tpr_pt,
               color=METHOD_COLORS[thr_lbl], s=40, zorder=5,
               edgecolors="black", lw=0.8,
               label=f"{thr_g}g")

# ── Reference diagonal ────────────────────────────────────────────────────────
ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.4)

# ── DeLong result printed on plot ─────────────────────────────────────────────
# All three threshold methods share max_acc_g, so DeLong z/p is identical;
# report once using Threshold_400g row.
r_dl = metrics_df[metrics_df["method"] == "Threshold_400g"].iloc[0]
if not pd.isna(r_dl.get("delong_z", np.nan)):
    delong_txt = (
        f"DeLong test\n"
        f"miniRocket AUC = {mr_auc_val:.3f}\n"
        f"acceleration θ AUC = {thr_auc_val:.3f}\n"
        f"z = {r_dl['delong_z']:.3f},  p = {r_dl['delong_p']:.4f}"
    )
    ax.text(0.98, 0.8, delong_txt,
            transform=ax.transAxes, fontsize=6.5,
            va="bottom", ha="right")

# ── Axes and spines ───────────────────────────────────────────────────────────
ax.set_xlabel("1 − Specificity", fontsize=9)
ax.set_ylabel("Sensitivity", fontsize=9)
ax.tick_params(labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_position(("outward", 4))
ax.spines["bottom"].set_position(("outward", 4))
ax.spines["left"].set_bounds(0, 1.0)
ax.spines["bottom"].set_bounds(0, 1.0)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(loc="lower right", fontsize=7, frameon=True,
          edgecolor="#cccccc", framealpha=0.95)

plt.savefig(f"{BINARY_DIR}/model_performance_combined.svg", bbox_inches="tight")
plt.savefig(f"{BINARY_DIR}/model_performance_combined.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved model_performance_combined")
