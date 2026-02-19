import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sktime.transformations.panel.rocket import MiniRocket
import matplotlib.pyplot as plt
import seaborn as sns


RAW_DATA = "python_data/labeled_data_with_types.csv"
OUTDIR   = "python_results/multiclass"
DATADIR  = "python_data"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)
np.random.seed(42)

# 0=no_contact | 1=other_impeller | 2=leading_indirect | 3=leading_direct
CLASS_NAMES = ['no_contact', 'other_impeller', 'leading_indirect', 'leading_direct']
N_CLASSES   = 4

print("=" * 60)
print("MULTICLASS BLADE STRIKE CLASSIFIER (4-class)")
print("Classes: 0=no_contact, 1=other_impeller, 2=leading_indirect, 3=leading_direct")
print("=" * 60)

df = pd.read_csv(RAW_DATA, low_memory=False)
print(f"Rows: {len(df)}")

def create_multiclass_label(row):
    if row["passage_type"] == "No contact":
        return 0
    elif row["passage_type"] == "Other impeller collision":
        return 1
    elif row["passage_type"] == "Leading edge strike" and row["leading_type"] == "Indirect":
        return 2
    elif row["passage_type"] == "Leading edge strike" and row["leading_type"] == "Direct":
        return 3
    return -1

df["strike_class"] = df.apply(create_multiclass_label, axis=1)
df["strike_type"]  = df["strike_class"].map(
    {0: "no_contact", 1: "other_impeller", 2: "leading_indirect", 3: "leading_direct"})

# Check for unmapped rows
unmapped = (df["strike_class"] == -1).sum()
if unmapped > 0:
    print(f"WARNING: {unmapped} rows could not be mapped to a class - check passage_type/leading_type values")

df = df[df["strike_class"] != -1].copy()

file_metadata = (
    df.groupby("file")
      .agg({
          "strike_class":     "first",
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
print(f"\nClass distribution:")
for i, cn in enumerate(CLASS_NAMES):
    count = (file_metadata['strike_class'] == i).sum()
    print(f"  Class {i} ({cn:20s}): {count:3d} ({count/len(file_metadata)*100:.1f}%)")

# ============================================================================
# Time series extraction
# ============================================================================
print("\n" + "=" * 60)
print("Extracting time series data")
print("=" * 60)

channels = [
    'higacc_x_g', 'higacc_y_g', 'higacc_z_g',
    'inacc_x_ms', 'inacc_y_ms', 'inacc_z_ms',
    'rot_x_degs', 'rot_y_degs', 'rot_z_degs',
    'pressure_kpa'
]

missing = [ch for ch in channels if ch not in df.columns]
if missing:
    print(f"Warning: Missing channels: {missing}")
    channels = [ch for ch in channels if ch in df.columns]
print(f"Using {len(channels)} channels")

def pad_time_series(ts, target_length):
    if len(ts) >= target_length:
        return ts[:target_length]
    n_pad = target_length - len(ts)
    return np.vstack([ts, np.tile(ts[-1], (n_pad, 1))])

lengths = []
for file_id in file_metadata['file']:
    lengths.append(len(df[df['file'] == file_id]))

max_length = max(lengths)
print(f"Sequence lengths: min={min(lengths)}, max={max_length}, mean={np.mean(lengths):.1f}")

X_list = []
for file_id in file_metadata['file']:
    file_data = df[df['file'] == file_id].sort_values('time_s')
    ts_padded = pad_time_series(file_data[channels].values, max_length)
    X_list.append(ts_padded.T)

X = np.stack(X_list)
y = file_metadata["strike_class"].values

print(f"\nArray shapes: X={X.shape}, y={y.shape}")

np.save(f"{DATADIR}/X_multiclass.npy", X)
np.save(f"{DATADIR}/y_multiclass.npy", y)
file_metadata.to_csv(f"{OUTDIR}/feature_metadata.csv", index=False)

# Class weights
class_weights_arr = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights_arr))
sample_weights_full = np.array([class_weight_dict[label] for label in y])

print(f"\nClass weights (balanced):")
for i, cn in enumerate(CLASS_NAMES):
    print(f"  {cn:22s}: {class_weight_dict[i]:.3f}")

# ============================================================================
# 5-Fold Stratified Cross-Validation
# ============================================================================
print("\n" + "=" * 60)
print("5-Fold Stratified Cross-Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds       = np.zeros(len(y), dtype=int)
oof_probs       = np.zeros((len(y), N_CLASSES))
fold_assignment = np.zeros(len(y), dtype=int)

cv_metrics = {'accuracy': [], 'macro_precision': [],
              'macro_recall': [], 'macro_f1': []}

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold+1}/5: train={len(train_idx)}, test={len(test_idx)}")

    pipeline = make_pipeline(
        MiniRocket(random_state=42, n_jobs=-1),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')
    )

    fold_weights = sample_weights_full[train_idx]
    pipeline.fit(X[train_idx], y[train_idx],
                 ridgeclassifiercv__sample_weight=fold_weights)

    scores = pipeline.decision_function(X[test_idx])  # (n_test, 4)
    exp_s  = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs  = exp_s / exp_s.sum(axis=1, keepdims=True)
    preds  = np.argmax(scores, axis=1)

    oof_preds[test_idx]       = preds
    oof_probs[test_idx]       = probs
    fold_assignment[test_idx] = fold

    acc_f = accuracy_score(y[test_idx], preds)
    prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(
        y[test_idx], preds, average='macro', zero_division=0)

    cv_metrics['accuracy'].append(acc_f)
    cv_metrics['macro_precision'].append(prec_f)
    cv_metrics['macro_recall'].append(rec_f)
    cv_metrics['macro_f1'].append(f1_f)

    print(f"  Accuracy={acc_f:.3f}, Macro-F1={f1_f:.3f}")

# ============================================================================
# Overall out-of-fold performance
# ============================================================================
print("\n" + "=" * 60)
print("Overall out-of-fold performance")
print("=" * 60)

overall_accuracy = accuracy_score(y, oof_preds)
prec_all, rec_all, f1_all, support_all = precision_recall_fscore_support(
    y, oof_preds, average=None, zero_division=0)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y, oof_preds, average='macro', zero_division=0)

cm = confusion_matrix(y, oof_preds)

print(f"\nOut-of-fold performance:")
print(f"  Overall Accuracy:  {overall_accuracy:.3f}")
print(f"  Macro Precision:   {prec_macro:.3f}")
print(f"  Macro Recall:      {rec_macro:.3f}")
print(f"  Macro F1-Score:    {f1_macro:.3f}")

print(f"\nPer-class performance:")
print(f"{'Class':22s} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 62)
for i, cn in enumerate(CLASS_NAMES):
    print(f"{cn:22s} {prec_all[i]:>10.3f} {rec_all[i]:>10.3f} {f1_all[i]:>10.3f} {support_all[i]:>10d}")

print(f"\nCV summary (mean ± std, 5 folds):")
for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']:
    print(f"  {k:20s}: {np.mean(cv_metrics[k]):.3f} ± {np.std(cv_metrics[k]):.3f}")

print(f"\nConfusion Matrix (rows=true, cols=predicted):")
header = f"{'':22s}" + "".join([f"{cn:>18s}" for cn in CLASS_NAMES])
print(header)
for i, cn in enumerate(CLASS_NAMES):
    row_str = f"{cn:22s}" + "".join([f"{cm[i,j]:>18d}" for j in range(N_CLASSES)])
    print(row_str)

# ============================================================================
# Misclassified files
# ============================================================================
print("\n" + "=" * 60)
print("Misclassified files for review")
print("=" * 60)

cv_results = file_metadata.copy()
cv_results["y_true"]      = y
cv_results["y_pred"]      = oof_preds
cv_results["true_class"]  = [CLASS_NAMES[t] for t in y]
cv_results["pred_class"]  = [CLASS_NAMES[p] for p in oof_preds]
cv_results["cv_fold"]     = fold_assignment
cv_results["correct"]     = (cv_results["y_pred"] == cv_results["y_true"])
cv_results["confidence"]  = oof_probs.max(axis=1)
for i, cn in enumerate(CLASS_NAMES):
    cv_results[f"prob_{cn}"] = oof_probs[:, i]

misclassified = cv_results[~cv_results["correct"]].copy()

print(f"\nTotal misclassified: {len(misclassified)}")
print(f"\nMisclassification breakdown (true -> predicted):")
breakdown = (misclassified
             .groupby(["true_class", "pred_class"])
             .size()
             .reset_index(name="count")
             .sort_values("count", ascending=False))
print(breakdown.to_string(index=False))

print(f"\nPer-class misclassification:")
for i, cn in enumerate(CLASS_NAMES):
    true_mask = cv_results["y_true"] == i
    n_total   = true_mask.sum()
    n_wrong   = (~cv_results.loc[true_mask, "correct"]).sum()
    print(f"  {cn:22s}: {n_wrong}/{n_total} misclassified ({n_wrong/n_total*100:.1f}%)")

# ============================================================================
# Strike rate and severity decomposition by treatment
# ============================================================================
print("\n" + "=" * 60)
print("Strike rate and severity decomposition (Wilson 95% CI)")
print("=" * 60)

def wilson_ci(k, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    z     = 1.959964
    p_hat = k / n
    denom  = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return p_hat, max(0, centre - margin), min(1, centre + margin)

rows = []
for tx, grp in cv_results.groupby("treatment"):
    n = len(grp)
    n_any_strike    = int((grp["y_pred"] > 0).sum())
    n_severe        = int((grp["y_pred"] == 3).sum())         # leading_direct
    n_less_severe   = int((grp["y_pred"].isin([1, 2])).sum()) # other + leading_indirect
    n_true_strike   = int((grp["y_true"] > 0).sum())

    strike_rate, sr_lo, sr_hi     = wilson_ci(n_any_strike, n)
    severe_rate, sev_lo, sev_hi   = wilson_ci(n_severe, n)
    less_rate, less_lo, less_hi   = wilson_ci(n_less_severe, n)

    # Proportion of strikes that are severe (conditional)
    sev_prop = n_severe / n_any_strike if n_any_strike > 0 else 0.0

    rows.append({
        "treatment":          tx,
        "n_fish":             n,
        "n_true_strike":      n_true_strike,
        "n_pred_strike":      n_any_strike,
        "strike_rate":        round(strike_rate, 4),
        "strike_ci_lo":       round(sr_lo, 4),
        "strike_ci_hi":       round(sr_hi, 4),
        "n_severe":           n_severe,
        "severe_rate":        round(severe_rate, 4),
        "severe_ci_lo":       round(sev_lo, 4),
        "severe_ci_hi":       round(sev_hi, 4),
        "n_less_severe":      n_less_severe,
        "less_severe_rate":   round(less_rate, 4),
        "less_severe_ci_lo":  round(less_lo, 4),
        "less_severe_ci_hi":  round(less_hi, 4),
        "prop_severe_of_strikes": round(sev_prop, 4)
    })

severity_summary = pd.DataFrame(rows)
print(severity_summary.to_string(index=False))

# ============================================================================
# Figures
# ============================================================================
plt.style.use('default')
class_colors = ['steelblue', 'darkorange', 'tomato', 'seagreen']

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(9, 7))
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
annot  = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(N_CLASSES)]
                    for i in range(N_CLASSES)])
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={'size': 10}, ax=ax)
ax.set_title(f'Confusion Matrix (Accuracy = {overall_accuracy:.3f})',
             fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Per-class recall bar chart
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(CLASS_NAMES, rec_all, color=class_colors, edgecolor='black', width=0.6)
ax.axhline(y=overall_accuracy, color='black', linestyle='--', lw=1.5,
           label=f'Overall Accuracy ({overall_accuracy:.3f})')
ax.set_ylabel('Recall', fontsize=12)
ax.set_xlabel('Class', fontsize=12)
ax.set_title('Per-Class Recall', fontsize=13, fontweight='bold')
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, rec_all):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/per_class_recall.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Prediction confidence by true class
fig, axes = plt.subplots(1, N_CLASSES, figsize=(16, 5))
for i, cn in enumerate(CLASS_NAMES):
    ax         = axes[i]
    class_mask = (y == i)
    ax.hist(oof_probs[class_mask, i], bins=20, alpha=0.75,
            edgecolor='black', color=class_colors[i])
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'True: {cn}\n(n={class_mask.sum()})', fontsize=11, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(alpha=0.3)
plt.suptitle('Prediction Confidence by True Class', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/prediction_confidence.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Severity decomposition by treatment
fig, ax = plt.subplots(figsize=(10, 6))
x      = np.arange(len(severity_summary))
width  = 0.35
ax.bar(x - width/2, severity_summary["severe_rate"],    width, label='Severe (leading_direct)',
       color='tomato', edgecolor='black')
ax.bar(x + width/2, severity_summary["less_severe_rate"], width, label='Less severe (other + leading_indirect)',
       color='steelblue', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(severity_summary["treatment"], rotation=45, ha='right')
ax.set_ylabel('Predicted Strike Rate', fontsize=12)
ax.set_xlabel('Treatment', fontsize=12)
ax.set_title('Strike Severity Decomposition by Treatment', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/severity_by_treatment.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved figures")

# ============================================================================
# Save outputs
# ============================================================================
cv_results.to_csv(f"{OUTDIR}/cv_predictions.csv", index=False)
misclassified.to_csv(f"{OUTDIR}/misclassified_files_for_review.csv", index=False)
severity_summary.to_csv(f"{OUTDIR}/severity_summary.csv", index=False)

metrics = {
    "model": "MiniRocket + RidgeClassifierCV (4-class)",
    "n_samples":      int(len(y)),
    "n_classes":      N_CLASSES,
    "class_names":    CLASS_NAMES,
    "class_distribution": {CLASS_NAMES[i]: int((y == i).sum()) for i in range(N_CLASSES)},
    "class_weights":      {CLASS_NAMES[i]: float(class_weight_dict[i]) for i in range(N_CLASSES)},
    "n_channels":          len(channels),
    "channels":            channels,
    "max_sequence_length": int(max_length),
    "cross_validation": {
        "n_folds": 5,
        **{f"mean_{k}": float(np.mean(v)) for k, v in cv_metrics.items()},
        **{f"std_{k}":  float(np.std(v))  for k, v in cv_metrics.items()}
    },
    "out_of_fold_performance": {
        "overall_accuracy": float(overall_accuracy),
        "macro_precision":  float(prec_macro),
        "macro_recall":     float(rec_macro),
        "macro_f1":         float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            CLASS_NAMES[i]: {
                "precision": float(prec_all[i]),
                "recall":    float(rec_all[i]),
                "f1_score":  float(f1_all[i]),
                "support":   int(support_all[i])
            } for i in range(N_CLASSES)
        }
    },
    "misclassified": {
        "total": int(len(misclassified)),
        "by_true_class": {
            CLASS_NAMES[i]: int((misclassified["y_true"] == i).sum()) for i in range(N_CLASSES)
        }
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
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')
)
final_pipeline.fit(X, y, ridgeclassifiercv__sample_weight=sample_weights_full)
joblib.dump(final_pipeline, f"{OUTDIR}/final_model_for_deployment.joblib")
np.save(f"{OUTDIR}/max_sequence_length.npy", max_length)
print("Saved final_model_for_deployment.joblib")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\n  Files processed:   {len(y)}")
print(f"  Overall Accuracy:  {overall_accuracy:.3f}")
print(f"  Macro F1-Score:    {f1_macro:.3f}")
print(f"\n  Per-class recall:")
for i, cn in enumerate(CLASS_NAMES):
    print(f"    {cn:22s}: {rec_all[i]:.3f}  (n={support_all[i]})")
print(f"\n  Misclassified:     {len(misclassified)}")
print(f"\nOutputs: {OUTDIR}/")
