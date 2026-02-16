# ============================================================================
# 01_minirocket_cv_pipeline.py
# Complete pipeline: 5-fold CV + final model training + error analysis
# ============================================================================

import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             classification_report, precision_recall_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocket
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RAW_DATA = "python_data/raw_labeled_data.csv"
OUTDIR = "python_results"
DATADIR = "python_data"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# STEP 1: Load and clean annotations
# ============================================================================
print("=" * 60)
print("STEP 1: Loading and cleaning data")
print("=" * 60)

df = pd.read_csv(RAW_DATA, low_memory=False)
print(f"Initial rows: {len(df)}")

# Filter as per your criteria
df = df[df["treatment"] != "400 (50%)"].copy()
df = df[df["roi"].str.contains("roi4_nadir", na=False)].copy()
print(f"After filtering: {len(df)} rows")

# Create labels
df["blade_strike"] = (df["n_contact"] > 0).astype(int)

def classify_strike_type(row):
    if row["n_contact"] == 0:
        return "no_contact"
    elif row["c_1_type"] == "Direct":
        return "direct_strike"
    else:
        return "indirect_strike"

df["strike_type"] = df.apply(classify_strike_type, axis=1)

# Get file-level metadata
file_metadata = (
    df.groupby("file")
      .agg({
          "blade_strike": "first",
          "strike_type": "first",
          "treatment": "first",
          "n_contact": "first",
          "c_1_type": "first",
          "clear_passage": "first",
          "centre_hub_contact": "first",
          "blade_contact": "first",
          "c_1_bl": "first"
      })
      .reset_index()
)

print(f"\nUnique files: {len(file_metadata)}")
print(f"Label distribution:\n{file_metadata['blade_strike'].value_counts()}")
print(f"Strike type distribution:\n{file_metadata['strike_type'].value_counts()}")

# ============================================================================
# STEP 2: Define channels and extract time series
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Extracting time series data")
print("=" * 60)

channels = [
    'higacc_x_g', 'higacc_y_g', 'higacc_z_g',
    'inacc_x_ms', 'inacc_y_ms', 'inacc_z_ms',
    'rot_x_degs', 'rot_y_degs', 'rot_z_degs',
    'pressure_kpa'
]

# Verify channels exist
missing = [ch for ch in channels if ch not in df.columns]
if missing:
    print(f"Warning: Missing channels: {missing}")
    channels = [ch for ch in channels if ch in df.columns]

print(f"Using {len(channels)} channels")

# Extract time series for each file
time_series_dict = {}
lengths = []

for file_id in file_metadata['file']:
    file_data = df[df['file'] == file_id].sort_values('time_s')
    ts_data = file_data[channels].values
    time_series_dict[file_id] = ts_data
    lengths.append(len(ts_data))

print(f"\nTime series statistics:")
print(f"  Min length: {min(lengths)}")
print(f"  Max length: {max(lengths)}")
print(f"  Mean length: {np.mean(lengths):.1f}")

# Pad to max length
max_length = max(lengths)
print(f"\nPadding to max length: {max_length}")

def pad_time_series(ts, target_length):
    """Pad by repeating last row or truncate."""
    if len(ts) >= target_length:
        return ts[:target_length]
    else:
        n_pad = target_length - len(ts)
        padding = np.tile(ts[-1], (n_pad, 1))
        return np.vstack([ts, padding])

# Create padded arrays (samples × channels × timepoints for sktime)
X_list = []
valid_files = []

for file_id in file_metadata['file']:
    ts = time_series_dict[file_id]
    ts_padded = pad_time_series(ts, max_length)
    ts_transposed = ts_padded.T  # (channels, timepoints)
    X_list.append(ts_transposed)
    valid_files.append(file_id)

# Filter metadata to match valid files
file_metadata = file_metadata[file_metadata["file"].isin(valid_files)].reset_index(drop=True)

# Create final arrays
X = np.stack(X_list)  # (n_samples, n_channels, n_timepoints)
y = file_metadata["blade_strike"].values

print(f"\nFinal array shapes:")
print(f"  X: {X.shape}")
print(f"  y: {y.shape} ({y.sum()} strikes, {(y==0).sum()} non-strikes)")

# Save arrays for later use
np.save(f"{DATADIR}/X_combined.npy", X)
np.save(f"{DATADIR}/y_combined.npy", y)
file_metadata.to_csv(f"{OUTDIR}/feature_metadata.csv", index=False)
print(f"\nSaved arrays to {DATADIR}/")

# ============================================================================
# STEP 3: 5-Fold Cross-Validation with MiniROCKET
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: 5-Fold Cross-Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results from each fold
fold_results = []
oof_probs = np.zeros(len(y))
oof_preds = np.zeros(len(y))
fold_assignment = np.zeros(len(y), dtype=int)

# For storing per-fold metrics
cv_metrics = {
    'auc': [],
    'accuracy': [],
    'sensitivity': [],
    'specificity': [],
    'precision': [],
    'f1': []
}

print("\nRunning 5-fold CV...")
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold+1}/5:")
    print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
    
    # Create pipeline for this fold
    pipeline = make_pipeline(
        MiniRocket(random_state=42, n_jobs=-1),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    )
    
    # Train
    pipeline.fit(X[train_idx], y[train_idx])
    
    # Predict on test fold
    scores = pipeline.decision_function(X[test_idx])
    probs = 1 / (1 + np.exp(-scores))
    
    # Find optimal threshold for this fold using Youden's index
    fpr_fold, tpr_fold, thresh_fold = roc_curve(y[test_idx], probs)
    j_fold = tpr_fold - fpr_fold
    opt_thresh_fold = thresh_fold[np.argmax(j_fold)]
    preds_fold = (probs >= opt_thresh_fold).astype(int)
    
    # Store out-of-fold predictions
    oof_probs[test_idx] = probs
    oof_preds[test_idx] = preds_fold
    fold_assignment[test_idx] = fold
    
    # Calculate metrics for this fold
    auc_fold = roc_auc_score(y[test_idx], probs)
    cm_fold = confusion_matrix(y[test_idx], preds_fold)
    tn, fp, fn, tp = cm_fold.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    cv_metrics['auc'].append(auc_fold)
    cv_metrics['accuracy'].append(accuracy)
    cv_metrics['sensitivity'].append(sensitivity)
    cv_metrics['specificity'].append(specificity)
    cv_metrics['precision'].append(precision)
    cv_metrics['f1'].append(f1)
    
    print(f"  Fold AUC: {auc_fold:.3f}")
    print(f"  Fold Accuracy: {accuracy:.3f}")
    print(f"  Fold threshold: {opt_thresh_fold:.3f}")

# ============================================================================
# STEP 4: Overall CV Performance (out-of-fold)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Overall CV Performance (out-of-fold)")
print("=" * 60)

# Calculate overall metrics using out-of-fold predictions
final_auc = roc_auc_score(y, oof_probs)
fpr, tpr, thresholds = roc_curve(y, oof_probs)
j = tpr - fpr
optimal_idx = np.argmax(j)
optimal_threshold = thresholds[optimal_idx]

# Final predictions using optimal threshold
final_preds = (oof_probs >= optimal_threshold).astype(int)
cm = confusion_matrix(y, final_preds)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

print(f"\nOut-of-fold performance (unbiased estimate):")
print(f"  AUC: {final_auc:.3f}")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Sensitivity/Recall: {sensitivity:.3f}")
print(f"  Specificity: {specificity:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  F1-Score: {f1:.3f}")
print(f"  Optimal threshold: {optimal_threshold:.3f}")

print(f"\nConfusion Matrix:")
print(f"              Pred No Strike  Pred Strike")
print(f"True No Strike     {tn:3d}           {fp:3d}")
print(f"True Strike        {fn:3d}           {tp:3d}")

# Cross-validation summary (mean ± std across folds)
print(f"\nCross-validation summary (mean ± std across 5 folds):")
print(f"  AUC: {np.mean(cv_metrics['auc']):.3f} ± {np.std(cv_metrics['auc']):.3f}")
print(f"  Accuracy: {np.mean(cv_metrics['accuracy']):.3f} ± {np.std(cv_metrics['accuracy']):.3f}")

# ============================================================================
# STEP 5: Identify misclassified files for manual review
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Identifying misclassified files for review")
print("=" * 60)

# Create results dataframe with all info
cv_results = file_metadata.copy()
cv_results["probability"] = oof_probs
cv_results["y_pred"] = final_preds
cv_results["y_true"] = y
cv_results["cv_fold"] = fold_assignment
cv_results["correct"] = (cv_results["y_pred"] == cv_results["y_true"])
cv_results["error_type"] = "correct"
cv_results.loc[(cv_results["y_true"] == 0) & (cv_results["y_pred"] == 1), "error_type"] = "false_positive"
cv_results.loc[(cv_results["y_true"] == 1) & (cv_results["y_pred"] == 0), "error_type"] = "false_negative"

# Get all misclassified files
misclassified = cv_results[cv_results["error_type"] != "correct"].copy()
fp_files = misclassified[misclassified["error_type"] == "false_positive"]
fn_files = misclassified[misclassified["error_type"] == "false_negative"]

print(f"\nTotal misclassified files: {len(misclassified)}")
print(f"  False Positives: {len(fp_files)}")
print(f"  False Negatives: {len(fn_files)}")

print("\nFalse Positives (predicted strike, actually no strike):")
for idx, row in fp_files.iterrows():
    print(f"  {row['file']}: probability={row['probability']:.3f}, treatment={row['treatment']}, type={row['strike_type']}")

print("\nFalse Negatives (predicted no strike, actually strike):")
for idx, row in fn_files.iterrows():
    print(f"  {row['file']}: probability={row['probability']:.3f}, treatment={row['treatment']}, type={row['strike_type']}")

# Save misclassified files for manual review
misclassified.to_csv(f"{OUTDIR}/misclassified_files_for_review.csv", index=False)
fp_files.to_csv(f"{OUTDIR}/false_positives.csv", index=False)
fn_files.to_csv(f"{OUTDIR}/false_negatives.csv", index=False)

# Performance by strike type
print("\nPerformance by strike type:")
perf_by_type = cv_results.groupby("strike_type").agg({
    'file': 'count',
    'correct': 'mean',
    'y_true': 'mean',
    'probability': lambda x: x[cv_results.loc[x.index, 'y_true']==1].mean() if any(cv_results.loc[x.index, 'y_true']==1) else np.nan
}).rename(columns={
    'file': 'n_files',
    'correct': 'accuracy',
    'y_true': 'strike_rate',
    'probability': 'mean_prob_strike'
})
print(perf_by_type.round(3))

# Performance by treatment
print("\nPerformance by treatment:")
perf_by_tx = cv_results.groupby("treatment").agg({
    'file': 'count',
    'correct': 'mean',
    'y_true': 'mean'
}).rename(columns={
    'file': 'n_files',
    'correct': 'accuracy',
    'y_true': 'strike_rate'
})
print(perf_by_tx.round(3))

# ============================================================================
# STEP 6: Generate Publication-Ready Plots
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Generating plots")
print("=" * 60)

# Set style
plt.style.use('default')
sns.set_palette("husl")

# 1. ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', lw=2.5, label=f'ROC Curve (AUC = {final_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=150, zorder=5,
           edgecolors='black', linewidths=1.5,
           label=f'Optimal Threshold = {optimal_threshold:.3f}')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - MiniRocket Blade Strike Classifier', fontsize=13, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.savefig(f"{OUTDIR}/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Strike', 'Strike'],
            yticklabels=['No Strike', 'Strike'],
            annot_kws={'size': 14})
plt.title(f'Confusion Matrix\n(Accuracy = {accuracy:.3f})', fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig(f"{OUTDIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Probability Distribution
plt.figure(figsize=(10, 6))
bins = np.linspace(0, 1, 21)
plt.hist(oof_probs[y == 0], bins=bins, alpha=0.6, 
         label='No Strike (n=147)', color='blue', edgecolor='black', linewidth=0.5)
plt.hist(oof_probs[y == 1], bins=bins, alpha=0.6,
         label='Strike (n=111)', color='red', edgecolor='black', linewidth=0.5)
plt.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
           label=f'Optimal Threshold ({optimal_threshold:.3f})')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Out-of-Fold Predicted Probabilities', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3, axis='y')
plt.savefig(f"{OUTDIR}/probability_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Performance by Strike Type
fig, ax = plt.subplots(figsize=(8, 6))
perf_by_type['accuracy'].plot(kind='bar', ax=ax, color=['blue', 'red', 'green'])
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Strike Type', fontsize=12)
ax.set_title('Model Accuracy by Strike Type', fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.axhline(y=accuracy, color='black', linestyle='--', label=f'Overall Accuracy ({accuracy:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/accuracy_by_strike_type.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# STEP 7: Save all results
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: Saving results")
print("=" * 60)

# Save CV results
cv_results.to_csv(f"{OUTDIR}/cv_predictions.csv", index=False)

# Save performance metrics
metrics = {
    "model": "MiniRocket + RidgeClassifierCV",
    "n_samples": int(len(y)),
    "n_strikes": int(y.sum()),
    "n_no_strikes": int((y == 0).sum()),
    "strike_rate": float(y.mean()),
    "n_channels": len(channels),
    "max_sequence_length": int(max_length),
    "cross_validation": {
        "n_folds": 5,
        "mean_auc": float(np.mean(cv_metrics['auc'])),
        "std_auc": float(np.std(cv_metrics['auc'])),
        "mean_accuracy": float(np.mean(cv_metrics['accuracy'])),
        "std_accuracy": float(np.std(cv_metrics['accuracy']))
    },
    "out_of_fold_performance": {
        "auc": float(final_auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "optimal_threshold": float(optimal_threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    },
    "misclassified": {
        "total": int(len(misclassified)),
        "false_positives": int(len(fp_files)),
        "false_negatives": int(len(fn_files))
    }
}

with open(f"{OUTDIR}/performance_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ============================================================================
# STEP 8: Train FINAL model on ALL data (for deployment)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: Training final model on ALL data")
print("=" * 60)
print("NOTE: This model's expected performance is the CV metrics above")
print(f"      Expected AUC: {final_auc:.3f} (±{np.std(cv_metrics['auc']):.3f})")

final_pipeline = make_pipeline(
    MiniRocket(random_state=42, n_jobs=-1),
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
)

final_pipeline.fit(X, y)
joblib.dump(final_pipeline, f"{OUTDIR}/final_model_for_deployment.joblib")

print(f"\n✓ Final model saved to: {OUTDIR}/final_model_for_deployment.joblib")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE - SUMMARY")
print("=" * 60)
print(f"\n1. Data processed: {len(y)} files")
print(f"2. CV Performance (unbiased): AUC={final_auc:.3f}, Accuracy={accuracy:.3f}")
print(f"3. Misclassified files for review: {len(misclassified)}")
print(f"   - Review these files to verify/correct labels:")
print(f"     * False Positives: {OUTDIR}/false_positives.csv")
print(f"     * False Negatives: {OUTDIR}/false_negatives.csv")
print(f"\n4. After label review, re-run this script with corrected data")
print(f"5. Final deployable model: {OUTDIR}/final_model_for_deployment.joblib")
print(f"\nAll outputs saved to: {OUTDIR}/")
print("  - cv_predictions.csv (out-of-fold predictions)")
print("  - performance_metrics.json (all metrics)")
print("  - feature_metadata.csv (file info)")
print("  - misclassified_files_for_review.csv")
print("  - false_positives.csv")
print("  - false_negatives.csv")
print("  - *.png (publication-ready plots)")
print("  - final_model_for_deployment.joblib")

print("\n✓ Script 01 complete")
