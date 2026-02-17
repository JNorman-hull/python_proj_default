
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve, precision_recall_curve, auc)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sktime.transformations.panel.rocket import MiniRocket
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta as beta_dist


RAW_DATA = "python_data/raw_labeled_data2.csv"
OUTDIR   = "python_results/multiclass"
DATADIR  = "python_data"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)
np.random.seed(42)

CLASS_NAMES = ['no_contact', 'indirect_strike', 'direct_strike']


print("Classes: 0=no_contact, 1=indirect_strike, 2=direct_strike")

df = pd.read_csv(RAW_DATA, low_memory=False)
print(f"Initial rows: {len(df)}")

df = df[df["treatment"] != "400 (50%)"].copy()
df = df[df["centre_hub_contact"] != "CH contact"].copy()
df = df[df["roi"].str.contains("roi4_nadir", na=False)].copy()
print(f"After filtering: {len(df)} rows")

def create_multiclass_label(row):
    if row["n_contact"] == 0:
        return 0
    elif row["c_1_type"] == "Direct":
        return 2
    else:
        return 1

df["strike_class"] = df.apply(create_multiclass_label, axis=1)
df["strike_type"]  = df["strike_class"].map(
    {0: "no_contact", 1: "indirect_strike", 2: "direct_strike"})

file_metadata = (
    df.groupby("file")
      .agg({
          "strike_class": "first",
          "strike_type":  "first",
          "treatment":    "first",
          "n_contact":    "first",
          "c_1_type":     "first",
          "clear_passage":      "first",
          "centre_hub_contact": "first",
          "blade_contact":      "first",
          "c_1_bl":             "first"
      })
      .reset_index()
)

print(f"\nUnique files: {len(file_metadata)}")
print(f"\nClass distribution:")
for i, cn in enumerate(CLASS_NAMES):
    count = (file_metadata['strike_class'] == i).sum()
    print(f"  Class {i} ({cn:17s}): {count:3d} ({count/len(file_metadata)*100:.1f}%)")

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

time_series_dict = {}
lengths = []
for file_id in file_metadata['file']:
    file_data = df[df['file'] == file_id].sort_values('time_s')
    ts_data   = file_data[channels].values
    time_series_dict[file_id] = ts_data
    lengths.append(len(ts_data))

print(f"Time series lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

max_length = max(lengths)

def pad_time_series(ts, target_length):
    if len(ts) >= target_length:
        return ts[:target_length]
    n_pad = target_length - len(ts)
    return np.vstack([ts, np.tile(ts[-1], (n_pad, 1))])

X_list = []
for file_id in file_metadata['file']:
    ts_padded = pad_time_series(time_series_dict[file_id], max_length)
    X_list.append(ts_padded.T)

X = np.stack(X_list)
y = file_metadata["strike_class"].values

print(f"\nArray shapes: X={X.shape}, y={y.shape}")

np.save(f"{DATADIR}/X_multiclass.npy", X)
np.save(f"{DATADIR}/y_multiclass.npy", y)
file_metadata.to_csv(f"{OUTDIR}/feature_metadata.csv", index=False)

class_weights_arr = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights_arr))
sample_weights_full = np.array([class_weight_dict[label] for label in y])

print(f"\nClass weights (balanced):")
for i, cn in enumerate(CLASS_NAMES):
    print(f"  {cn:17s}: {class_weight_dict[i]:.3f}")

print("\n" + "=" * 60)
print("5-Fold Stratified Cross-Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds      = np.zeros(len(y), dtype=int)
oof_probs      = np.zeros((len(y), 3))  # softmax probabilities per class
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

    scores = pipeline.decision_function(X[test_idx])  # (n_test, 3)
    # Softmax probabilities need to chekc this is correct
    exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    preds = np.argmax(scores, axis=1)

    oof_preds[test_idx]      = preds
    oof_probs[test_idx]      = probs
    fold_assignment[test_idx] = fold

    acc_f = accuracy_score(y[test_idx], preds)
    prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(
        y[test_idx], preds, average='macro', zero_division=0)

    cv_metrics['accuracy'].append(acc_f)
    cv_metrics['macro_precision'].append(prec_f)
    cv_metrics['macro_recall'].append(rec_f)
    cv_metrics['macro_f1'].append(f1_f)

    print(f"  Accuracy={acc_f:.3f}, Macro-F1={f1_f:.3f}")


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
print(f"{'Class':20s} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 60)
for i, cn in enumerate(CLASS_NAMES):
    print(f"{cn:20s} {prec_all[i]:>10.3f} {rec_all[i]:>10.3f} {f1_all[i]:>10.3f} {support_all[i]:>10d}")

print(f"\nCV summary (mean ± std, 5 folds):")
for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']:
    print(f"  {k:18s}: {np.mean(cv_metrics[k]):.3f} ± {np.std(cv_metrics[k]):.3f}")

print(f"\nConfusion Matrix:")
print(f"                Pred NC  Pred Ind  Pred Dir")
print(f"True NC          {cm[0,0]:5d}    {cm[0,1]:5d}    {cm[0,2]:5d}")
print(f"True Indirect    {cm[1,0]:5d}    {cm[1,1]:5d}    {cm[1,2]:5d}")
print(f"True Direct      {cm[2,0]:5d}    {cm[2,1]:5d}    {cm[2,2]:5d}")


print("\n" + "=" * 60)
print("Misclassified files for review")
print("=" * 60)

cv_results = file_metadata.copy()
cv_results["y_true"]       = y
cv_results["y_pred"]       = oof_preds
cv_results["pred_class"]   = [CLASS_NAMES[p] for p in oof_preds]
cv_results["cv_fold"]      = fold_assignment
cv_results["correct"]      = (cv_results["y_pred"] == cv_results["y_true"])
# Add max probability (confidence)
cv_results["confidence"] = oof_probs.max(axis=1)
for i, cn in enumerate(CLASS_NAMES):
    cv_results[f"prob_{cn}"] = oof_probs[:, i]

misclassified = cv_results[~cv_results["correct"]].copy()
print(f"\nTotal misclassified: {len(misclassified)}")
print("\nMisclassification breakdown:")
for true_c in range(3):
    for pred_c in range(3):
        if true_c != pred_c:
            n = ((cv_results["y_true"] == true_c) & (cv_results["y_pred"] == pred_c)).sum()
            if n > 0:
                print(f"  True {CLASS_NAMES[true_c]:17s} → Pred {CLASS_NAMES[pred_c]:17s}: {n}")


print("\n" + "=" * 60)
print("Blade strike predictions per treatment")
print("=" * 60)

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.959964
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return p_hat, max(0, centre - margin), min(1, centre + margin)

# Predicted strike = class 1 or 2
cv_results["pred_strike"] = (cv_results["y_pred"] > 0).astype(int)
cv_results["true_strike"] = (cv_results["y_true"] > 0).astype(int)

rows = []
for tx, grp in cv_results.groupby("treatment"):
    n_total        = len(grp)
    n_pred_strike  = grp["pred_strike"].sum()
    n_true_strike  = grp["true_strike"].sum()
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


plt.style.use('default')
sns.set_palette("husl")
class_colors = ['steelblue', 'tomato', 'seagreen']

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(9, 7))
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
annot  = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(3)] for i in range(3)])
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
            xticklabels=['No Contact', 'Indirect', 'Direct'],
            yticklabels=['No Contact', 'Indirect', 'Direct'],
            annot_kws={'size': 11}, ax=ax)
ax.set_title(f'Confusion Matrix - Multi-Class\n(Overall Accuracy = {overall_accuracy:.3f})',
             fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Per-class performance bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(CLASS_NAMES))
w = 0.25
ax.bar(x - w, prec_all, w, label='Precision', alpha=0.8, edgecolor='black')
ax.bar(x,     rec_all,  w, label='Recall',    alpha=0.8, edgecolor='black')
ax.bar(x + w, f1_all,   w, label='F1-Score',  alpha=0.8, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Class', fontsize=12)
ax.set_title('Per-Class Performance Metrics', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/per_class_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. ROC Curves (one-vs-rest)
y_bin = label_binarize(y, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(8, 6))
for i, cn in enumerate(CLASS_NAMES):
    try:
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, i], oof_probs[:, i])
        auc_c = roc_auc_score(y_bin[:, i], oof_probs[:, i])
        ax.plot(fpr_c, tpr_c, lw=2, label=f'{cn} (AUC={auc_c:.3f})', color=class_colors[i])
    except Exception as e:
        print(f"  Warning: Could not plot ROC for {cn}: {e}")
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Multi-Class (One-vs-Rest)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Precision-Recall Curves (one-vs-rest)
fig, ax = plt.subplots(figsize=(8, 6))
for i, cn in enumerate(CLASS_NAMES):
    try:
        prec_c, rec_c, _ = precision_recall_curve(y_bin[:, i], oof_probs[:, i])
        pr_auc_c = auc(rec_c, prec_c)
        ax.plot(rec_c, prec_c, lw=2, label=f'{cn} (AUC={pr_auc_c:.3f})', color=class_colors[i])
    except Exception as e:
        print(f"  Warning: Could not plot PR curve for {cn}: {e}")
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Multi-Class (One-vs-Rest)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/precision_recall_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Prediction confidence (probability of predicted class)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, cn in enumerate(CLASS_NAMES):
    ax = axes[i]
    class_mask  = (y == i)
    class_probs = oof_probs[class_mask, i]
    ax.hist(class_probs, bins=20, alpha=0.75, edgecolor='black', color=class_colors[i])
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'True {cn}\n(n={class_mask.sum()})', fontsize=11, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(alpha=0.3)
plt.suptitle('Prediction Confidence by True Class', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/prediction_confidence.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Probability distributions (all 3 classes, per true class)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for true_c in range(3):
    mask = (y == true_c)
    for pred_c in range(3):
        ax = axes[true_c][pred_c]
        ax.hist(oof_probs[mask, pred_c], bins=20, alpha=0.75,
                edgecolor='black', color=class_colors[pred_c])
        ax.set_xlim([0, 1])
        ax.set_title(f'True={CLASS_NAMES[true_c]}\nP({CLASS_NAMES[pred_c]})',
                     fontsize=9)
        ax.grid(alpha=0.3)
plt.suptitle('Probability Distributions by True Class', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/probability_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

print("saved figures")

cv_results.to_csv(f"{OUTDIR}/cv_predictions.csv", index=False)
misclassified.to_csv(f"{OUTDIR}/misclassified_files_for_review.csv", index=False)
blade_strike_predictions.to_csv(f"{OUTDIR}/blade_strike_predictions.csv", index=False)

metrics = {
    "model": "MiniRocket + RidgeClassifierCV (Multi-Class)",
    "n_samples":   int(len(y)),
    "n_classes":   3,
    "class_names": CLASS_NAMES,
    "class_distribution": {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)},
    "class_weights":      {CLASS_NAMES[i]: float(class_weight_dict[i]) for i in range(3)},
    "n_channels":          len(channels),
    "max_sequence_length": int(max_length),
    "cross_validation": {
        "n_folds": 5,
        **{f"mean_{k}": float(np.mean(v)) for k, v in cv_metrics.items()},
        **{f"std_{k}":  float(np.std(v))  for k, v in cv_metrics.items()}
    },
    "out_of_fold_performance": {
        "overall_accuracy":  float(overall_accuracy),
        "macro_precision":   float(prec_macro),
        "macro_recall":      float(rec_macro),
        "macro_f1":          float(f1_macro),
        "confusion_matrix":  cm.tolist(),
        "per_class_metrics": {
            CLASS_NAMES[i]: {
                "precision": float(prec_all[i]),
                "recall":    float(rec_all[i]),
                "f1_score":  float(f1_all[i]),
                "support":   int(support_all[i])
            } for i in range(3)
        }
    },
    "misclassified": {
        "total": int(len(misclassified)),
        "by_true_class": {
            CLASS_NAMES[i]: int((misclassified["y_true"] == i).sum()) for i in range(3)
        }
    }
}

with open(f"{OUTDIR}/performance_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("saved csvs")

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
print(f"Saved final model for deployment")


print(f"\n  Files processed:   {len(y)}")
print(f"  Overall Accuracy:  {overall_accuracy:.3f}")
print(f"  Macro F1-Score:    {f1_macro:.3f}")
print(f"\n  Per-class recall:")
for i, cn in enumerate(CLASS_NAMES):
    print(f"    {cn:17s}: {rec_all[i]:.3f}")
print(f"\n  Misclassified:     {len(misclassified)}")
print(f"\nOutputs: {OUTDIR}/")
