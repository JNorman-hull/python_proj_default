import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             precision_recall_curve, auc)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocket
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta as beta_dist


RAW_DATA = "python_data/labeled_data_with_types.csv"
OUTDIR   = "python_results/binary"
DATADIR  = "python_data"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)
np.random.seed(42)

print("=" * 60)
print("BINARY BLADE STRIKE CLASSIFIER")
print("=" * 60)

df = pd.read_csv(RAW_DATA, low_memory=False)
df = df[df["passage_type"] != "Other impeller collision"].copy()
print(f"Rows: {len(df)}")

# Binary label
df["blade_strike"] = (df["passage_type"] != "No contact").astype(int)

# Strike type for downstream analysis
def classify_strike_type(row):
    if row["passage_type"] == "No contact":
        return "no_contact"
    elif row["passage_type"] == "Leading edge strike":
        return f"leading_{row['leading_type'].lower()}"
    elif row["passage_type"] == "Other impeller collision":
        return f"other_{row['other_type'].lower().replace(' ', '_')}"
    return "unknown"

df["strike_type"] = df.apply(classify_strike_type, axis=1)

# File-level metadata
file_metadata = (
    df.groupby("file")
      .agg({
          "blade_strike":     "first",
          "strike_type":      "first",
          "treatment":        "first",
          "passage_type":     "first",
          "leading_type":     "first",
          "other_type":       "first",
          "passage_severity": "first"
      })
      .reset_index()
)

print(f"\nUnique files: {len(file_metadata)}")
print(f"Strike distribution:")
print(file_metadata["strike_type"].value_counts())

# ============================================================================
# Time series extraction
# ============================================================================
channels = [
    'higacc_x_g', 'higacc_y_g', 'higacc_z_g',
    'inacc_x_ms', 'inacc_y_ms', 'inacc_z_ms',
    'rot_x_degs', 'rot_y_degs', 'rot_z_degs',
    'pressure_kpa'
]

def pad_time_series(ts, target_length):
    if len(ts) >= target_length:
        return ts[:target_length]
    n_pad = target_length - len(ts)
    return np.vstack([ts, np.tile(ts[-1], (n_pad, 1))])

lengths = []
for file_id in file_metadata['file']:
    file_data = df[df['file'] == file_id]
    lengths.append(len(file_data))

max_length = max(lengths)
print(f"\nSequence lengths: min={min(lengths)}, max={max_length}, mean={np.mean(lengths):.1f}")

X_list = []
for file_id in file_metadata['file']:
    file_data = df[df['file'] == file_id].sort_values('time_s')
    ts        = file_data[channels].values
    ts_padded = pad_time_series(ts, max_length)
    X_list.append(ts_padded.T)

X = np.stack(X_list)
y = file_metadata['blade_strike'].values

print(f"\nX shape: {X.shape}")
print(f"y distribution: {np.bincount(y)}")

# Class weights
n_strikes    = y.sum()
n_no_strikes = (y == 0).sum()
strike_weight = n_no_strikes / n_strikes

print(f"\nClass imbalance: {n_no_strikes} no-strikes, {n_strikes} strikes")
print(f"Strike class weight: {strike_weight:.2f}")

sample_weights_full = np.where(y == 1, strike_weight, 1.0)

# ============================================================================
# 5-Fold Stratified Cross-Validation
# ============================================================================
print("\n" + "=" * 60)
print("5-Fold Stratified Cross-Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probs       = np.zeros(len(y))
oof_preds       = np.zeros(len(y), dtype=int)
fold_assignment = np.zeros(len(y), dtype=int)

cv_metrics = {'auc': [], 'accuracy': [], 'sensitivity': [],
              'specificity': [], 'precision': [], 'f1': []}

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold+1}/5: train={len(train_idx)}, test={len(test_idx)}")

    pipeline = make_pipeline(
        MiniRocket(random_state=42, n_jobs=-1),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    )

    fold_weights = sample_weights_full[train_idx]
    pipeline.fit(X[train_idx], y[train_idx],
                 ridgeclassifiercv__sample_weight=fold_weights)

    scores = pipeline.decision_function(X[test_idx])
    probs  = 1 / (1 + np.exp(-scores))

    fpr_f, tpr_f, thresh_f = roc_curve(y[test_idx], probs)
    opt_thresh = thresh_f[np.argmax(tpr_f - fpr_f)]
    preds = (probs >= opt_thresh).astype(int)

    oof_probs[test_idx]       = probs
    oof_preds[test_idx]       = preds
    fold_assignment[test_idx] = fold

    auc_f = roc_auc_score(y[test_idx], probs)
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y[test_idx], preds).ravel()

    sens_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
    spec_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0
    prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
    acc_f  = (tp_f + tn_f) / (tp_f + tn_f + fp_f + fn_f)
    f1_f   = 2 * (prec_f * sens_f) / (prec_f + sens_f) if (prec_f + sens_f) > 0 else 0

    cv_metrics['auc'].append(auc_f)
    cv_metrics['accuracy'].append(acc_f)
    cv_metrics['sensitivity'].append(sens_f)
    cv_metrics['specificity'].append(spec_f)
    cv_metrics['precision'].append(prec_f)
    cv_metrics['f1'].append(f1_f)

    print(f"  AUC={auc_f:.3f}, Accuracy={acc_f:.3f}, Threshold={opt_thresh:.3f}")

# ============================================================================
# Overall out-of-fold performance
# ============================================================================
print("\n" + "=" * 60)
print("Overall out-of-fold performance")
print("=" * 60)

final_auc = roc_auc_score(y, oof_probs)
fpr, tpr, thresholds = roc_curve(y, oof_probs)
precision_curve, recall_curve, _ = precision_recall_curve(y, oof_probs)
pr_auc = auc(recall_curve, precision_curve)

optimal_idx       = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

final_preds = (oof_probs >= optimal_threshold).astype(int)
cm = confusion_matrix(y, final_preds)
tn, fp_n, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp_n)
precision   = tp / (tp + fp_n)
accuracy    = (tp + tn) / (tp + tn + fp_n + fn)
f1          = 2 * (precision * sensitivity) / (precision + sensitivity)

print(f"\nOut-of-fold performance:")
print(f"  AUC:               {final_auc:.3f}")
print(f"  PR-AUC:            {pr_auc:.3f}")
print(f"  Accuracy:          {accuracy:.3f}")
print(f"  Sensitivity:       {sensitivity:.3f}")
print(f"  Specificity:       {specificity:.3f}")
print(f"  Precision:         {precision:.3f}")
print(f"  F1-Score:          {f1:.3f}")
print(f"  Optimal threshold: {optimal_threshold:.3f}")

print(f"\nCV summary (mean ± std, 5 folds):")
for k in ['auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']:
    print(f"  {k:12s}: {np.mean(cv_metrics[k]):.3f} ± {np.std(cv_metrics[k]):.3f}")

print(f"\nConfusion Matrix:")
print(f"              Pred No Strike  Pred Strike")
print(f"True No Strike     {tn:3d}           {fp_n:3d}")
print(f"True Strike        {fn:3d}           {tp:3d}")

# ============================================================================
# Misclassified files
# ============================================================================
print("\n" + "=" * 60)
print("Misclassified files for review")
print("=" * 60)

cv_results = file_metadata.copy()
cv_results["probability"] = oof_probs
cv_results["y_pred"]      = final_preds
cv_results["y_true"]      = y
cv_results["cv_fold"]     = fold_assignment
cv_results["correct"]     = (cv_results["y_pred"] == cv_results["y_true"])
cv_results["error_type"]  = "correct"
cv_results.loc[(cv_results["y_true"] == 0) & (cv_results["y_pred"] == 1), "error_type"] = "false_positive"
cv_results.loc[(cv_results["y_true"] == 1) & (cv_results["y_pred"] == 0), "error_type"] = "false_negative"

misclassified = cv_results[cv_results["error_type"] != "correct"].copy()
fp_files      = misclassified[misclassified["error_type"] == "false_positive"]
fn_files      = misclassified[misclassified["error_type"] == "false_negative"]

print(f"\nTotal misclassified: {len(misclassified)}")
print(f"  False Positives: {len(fp_files)}")
print(f"  False Negatives: {len(fn_files)}")

print("\nFalse Positives:")
for _, row in fp_files.iterrows():
    print(f"  {row['file']}: prob={row['probability']:.3f}, treatment={row['treatment']}, type={row['strike_type']}")

print("\nFalse Negatives:")
for _, row in fn_files.iterrows():
    print(f"  {row['file']}: prob={row['probability']:.3f}, treatment={row['treatment']}, type={row['strike_type']}")

# ============================================================================
# Performance by strike type and treatment
# ============================================================================
print("\n" + "=" * 60)
print("Performance by strike type and treatment")
print("=" * 60)

print("\nPerformance by strike type:")
perf_by_type = (
    cv_results.groupby("strike_type")
    .agg(n_files=('file', 'count'),
         accuracy=('correct', 'mean'),
         mean_prob=('probability', 'mean'))
    .round(3)
)
print(perf_by_type)

print("\nPerformance by treatment:")
perf_by_tx = (
    cv_results.groupby("treatment")
    .agg(n_files=('file', 'count'),
         accuracy=('correct', 'mean'),
         strike_rate=('y_true', 'mean'))
    .round(3)
)
print(perf_by_tx)

# ============================================================================
# Blade strike predictions with Wilson CI
# ============================================================================
print("\n" + "=" * 60)
print("Blade strike predictions (Wilson binomial 95% CI)")
print("=" * 60)

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.959964
    p_hat = k / n
    denom  = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return p_hat, max(0, centre - margin), min(1, centre + margin)

tx_groups = cv_results.groupby("treatment")
rows = []
for tx, grp in tx_groups:
    n_total       = len(grp)
    n_pred_strike = grp["y_pred"].sum()
    n_true_strike = grp["y_true"].sum()
    p_hat, ci_lo, ci_hi = wilson_ci(n_pred_strike, n_total)
    rows.append({
        "treatment":             tx,
        "n_fish":                n_total,
        "n_predicted_strike":    int(n_pred_strike),
        "n_true_strike":         int(n_true_strike),
        "predicted_strike_rate": round(p_hat, 4),
        "wilson_ci_lower":       round(ci_lo, 4),
        "wilson_ci_upper":       round(ci_hi, 4),
        "accuracy":              round(grp["correct"].mean(), 4)
    })

blade_strike_predictions = pd.DataFrame(rows)
print(blade_strike_predictions.to_string(index=False))

# ============================================================================
# Figures
# ============================================================================
plt.style.use('default')
sns.set_palette("husl")

# 1. ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, 'b-', lw=2.5, label=f'ROC Curve (AUC = {final_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=150, zorder=5,
           edgecolors='black', lw=1.5,
           label=f'Optimal Threshold = {optimal_threshold:.3f}')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - MiniRocket Binary Classifier', fontsize=13, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Precision-Recall Curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall_curve, precision_curve, 'g-', lw=2.5,
        label=f'PR Curve (AUC = {pr_auc:.3f})')
ax.axhline(y=y.mean(), color='gray', linestyle='--', lw=1.5,
           label=f'Baseline (Strike Rate = {y.mean():.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve - MiniRocket Binary Classifier',
             fontsize=13, fontweight='bold')
ax.legend(loc="lower left", fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
annot  = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)])
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
            xticklabels=['No Strike', 'Strike'],
            yticklabels=['No Strike', 'Strike'],
            annot_kws={'size': 12}, ax=ax)
ax.set_title(f'Confusion Matrix (Accuracy = {accuracy:.3f})', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Probability Distribution
n_strikes_plot    = int(y.sum())
n_no_strikes_plot = int((y == 0).sum())
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(0, 1, 21)
ax.hist(oof_probs[y == 0], bins=bins, alpha=0.6,
        label=f'No Strike (n={n_no_strikes_plot})', color='steelblue', edgecolor='black', lw=0.5)
ax.hist(oof_probs[y == 1], bins=bins, alpha=0.6,
        label=f'Strike (n={n_strikes_plot})', color='tomato', edgecolor='black', lw=0.5)
ax.axvline(optimal_threshold, color='green', linestyle='--', lw=2,
           label=f'Optimal Threshold ({optimal_threshold:.3f})')
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Out-of-Fold Predicted Probabilities', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/probability_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Accuracy by Strike Type
fig, ax = plt.subplots(figsize=(9, 6))
perf_by_type['accuracy'].plot(kind='bar', ax=ax, edgecolor='black', width=0.6)
ax.axhline(y=accuracy, color='black', linestyle='--', lw=1.5,
           label=f'Overall Accuracy ({accuracy:.3f})')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Strike Type', fontsize=12)
ax.set_title('Model Accuracy by Strike Type', fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/accuracy_by_strike_type.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved figures")

# ============================================================================
# Save outputs
# ============================================================================
cv_results.to_csv(f"{OUTDIR}/cv_predictions.csv", index=False)
misclassified.to_csv(f"{OUTDIR}/misclassified_files_for_review.csv", index=False)
blade_strike_predictions.to_csv(f"{OUTDIR}/blade_strike_predictions.csv", index=False)

metrics = {
    "model": "MiniRocket + RidgeClassifierCV (Binary)",
    "n_samples":           int(len(y)),
    "n_strikes":           int(y.sum()),
    "n_no_strikes":        int((y == 0).sum()),
    "strike_rate":         float(y.mean()),
    "class_weight":        float(strike_weight),
    "n_channels":          len(channels),
    "max_sequence_length": int(max_length),
    "channels":            channels,
    "cross_validation": {
        "n_folds": 5,
        **{f"mean_{k}": float(np.mean(v)) for k, v in cv_metrics.items()},
        **{f"std_{k}":  float(np.std(v))  for k, v in cv_metrics.items()}
    },
    "out_of_fold_performance": {
        "roc_auc":           float(final_auc),
        "pr_auc":            float(pr_auc),
        "overall_accuracy":  float(accuracy),
        "sensitivity":       float(sensitivity),
        "specificity":       float(specificity),
        "precision":         float(precision),
        "f1_score":          float(f1),
        "optimal_threshold": float(optimal_threshold),
        "confusion_matrix":  {"tn": int(tn), "fp": int(fp_n), "fn": int(fn), "tp": int(tp)}
    },
    "performance_by_strike_type": {
        st: {
            "n_files":  int(grp["file"].count()),
            "accuracy": float(grp["correct"].mean())
        }
        for st, grp in cv_results.groupby("strike_type")
    },
    "misclassified": {
        "total":           int(len(misclassified)),
        "false_positives": int(len(fp_files)),
        "false_negatives": int(len(fn_files))
    }
}

with open(f"{OUTDIR}/performance_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved CSVs and metrics JSON")

# ============================================================================
# Train final model on ALL data
# ============================================================================
print("\n" + "=" * 60)
print("Training final model on ALL data")
print("=" * 60)

final_pipeline = make_pipeline(
    MiniRocket(random_state=42, n_jobs=-1),
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
)
final_pipeline.fit(X, y, ridgeclassifiercv__sample_weight=sample_weights_full)
joblib.dump(final_pipeline, f"{OUTDIR}/final_model_for_deployment.joblib")
print("Saved final_model_for_deployment.joblib")

# Save max_length for deployment
np.save(f"{OUTDIR}/max_sequence_length.npy", max_length)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\n  Files processed:  {len(y)}")
print(f"  ROC-AUC:          {final_auc:.3f}")
print(f"  Accuracy:         {accuracy:.3f}")
print(f"  Sensitivity:      {sensitivity:.3f}")
print(f"  Specificity:      {specificity:.3f}")
print(f"  Precision:        {precision:.3f}")
print(f"  F1-Score:         {f1:.3f}")
print(f"  Misclassified:    {len(misclassified)} files")
print(f"\nOutputs: {OUTDIR}/")
