import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

OUTDIR = "python_results/binary"

cv_results = pd.read_csv(f"{OUTDIR}/cv_predictions.csv")
with open(f"{OUTDIR}/performance_metrics.json") as f:
    metrics = json.load(f)

y           = cv_results["y_true"].values
oof_probs   = cv_results["probability"].values
final_preds = cv_results["y_pred"].values

oof               = metrics["out_of_fold_performance"]
final_auc         = oof["roc_auc"]
accuracy          = oof["overall_accuracy"]
sensitivity       = oof["sensitivity"]
specificity       = oof["specificity"]
mcc               = oof.get("mcc", None)
optimal_threshold = oof["optimal_threshold"]

fpr, tpr, thresholds = roc_curve(y, oof_probs)
optimal_idx = np.argmax(tpr - fpr)
cm = confusion_matrix(y, final_preds)

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16/2.54, 8/2.54),
                         gridspec_kw={'width_ratios': [1, 1]})

# ── Left: Confusion Matrix ────────────────────────────────────────────────────
ax = axes[0]
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
annot  = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)])
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
            xticklabels=['No Strike', 'Strike'],
            yticklabels=['No Strike', 'Strike'],
            annot_kws={'size': 9}, ax=ax,
            linewidths=0.5, linecolor='black',
            vmin=0, vmax=cm.max(),
            cbar=True,
            cbar_kws={"shrink": 0.8, "pad": 0.02})

# Black border on colorbar
cbar = ax.collections[0].colorbar
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
ax.set_ylabel('True Label', fontsize=8)
ax.set_xlabel('Predicted Label', fontsize=8)
ax.tick_params(labelsize=8)

# ── Right: ROC Curve ──────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {final_auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=60, zorder=5,
           edgecolors='black', lw=1,
           label=f'Threshold = {optimal_threshold:.3f}')
ax.set_xlabel('False Positive Rate', fontsize=8)
ax.set_ylabel('True Positive Rate', fontsize=8)
ax.tick_params(labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_position(("outward", 0.5))
ax.spines["bottom"].set_position(("outward", 0.5))
ax.spines["left"].set_bounds(0, 1.0)
ax.spines["bottom"].set_bounds(0, 1.0)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

# Stats box
lines = ["MiniRocket (OOF CV)",
         f"Accuracy :    {accuracy:.3f}",
         f"Sensitivity:  {sensitivity:.3f}",
         f"Specificity:  {specificity:.3f}",
         f"AUC :         {final_auc:.3f}"]
if mcc is not None:
    lines.append(f"MCC :         {mcc:.3f}")

ax.text(0.98, 0.1, "\n".join(lines),
        transform=ax.transAxes, fontsize=7,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#000000ff"))

ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/model_performance_combined.svg", bbox_inches='tight')
plt.savefig(f"{OUTDIR}/model_performance_combined.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved combined figure")
