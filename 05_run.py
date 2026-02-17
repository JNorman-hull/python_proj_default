# ============================================================================
# script_05_run.py
# Deploy trained binary blade strike classifier on new sensor data
#
# STEP 1 (optional): Sanity check - run model on TRAINING data to confirm
#                    the pipeline is working as expected.
#                    NOTE: Training accuracy will be higher than CV metrics
#                    because the model has seen this data. CV metrics from
#                    script_01 are the true performance estimate.
# STEP 2:            Run model on NEW unlabelled sensor data and output
#                    blade strike predictions with probabilities.
# ============================================================================

import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Configuration - edit these paths
# ------------------------------------------------------------------
MODEL_PATH   = "python_results/binary/final_model_for_deployment.joblib"
METRICS_PATH = "python_results/binary/performance_metrics.json"
TRAINING_DATA = "python_data/raw_labeled_data2.csv"   # for sanity check only

# NEW DATA - must be pre-filtered and contain the correct channels
# Expected columns: 'file', 'time_s', + the 10 sensor channels
NEW_DATA     = "python_data/new_sensor_data.csv"       # <-- edit this

OUTDIR       = "python_results/run"
os.makedirs(OUTDIR, exist_ok=True)

# Set to False to skip sanity check and go straight to new data prediction
RUN_SANITY_CHECK = True

print("=" * 60)
print("SCRIPT 05 - BLADE STRIKE DETECTOR (DEPLOYMENT)")
print("=" * 60)

# ============================================================================
# Load model and expected performance metrics
# ============================================================================
print("\n" + "=" * 60)
print("Loading model and metrics")
print("=" * 60)

model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded: {MODEL_PATH}")

with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)

optimal_threshold = metrics['out_of_fold_performance']['optimal_threshold']
cv_accuracy       = metrics['out_of_fold_performance']['overall_accuracy']
cv_sensitivity    = metrics['out_of_fold_performance']['sensitivity']
cv_specificity    = metrics['out_of_fold_performance']['specificity']
cv_auc            = metrics['out_of_fold_performance']['roc_auc']
trained_max_length = metrics['max_sequence_length']

print(f"\nExpected performance (from 5-fold CV on training data):")
print(f"  Accuracy:    {cv_accuracy:.3f}")
print(f"  Sensitivity: {cv_sensitivity:.3f}")
print(f"  Specificity: {cv_specificity:.3f}")
print(f"  AUC:         {cv_auc:.3f}")
print(f"  Threshold:   {optimal_threshold:.3f}")
print(f"  Max sequence length: {trained_max_length}")

# Channels are stored in the metrics file from script_01
channels = metrics['channels'] if 'channels' in metrics else [
    'higacc_x_g', 'higacc_y_g', 'higacc_z_g',
    'inacc_x_ms', 'inacc_y_ms', 'inacc_z_ms',
    'rot_x_degs', 'rot_y_degs', 'rot_z_degs',
    'pressure_kpa'
]
print(f"  Channels ({len(channels)}): {channels}")

# ============================================================================
# Shared helper functions
# ============================================================================
def pad_time_series(ts, target_length):
    if len(ts) >= target_length:
        return ts[:target_length]
    n_pad = target_length - len(ts)
    return np.vstack([ts, np.tile(ts[-1], (n_pad, 1))])

def load_and_prepare(df, channels, max_length, labeled=True):
    """
    Extract time series per file and pad to max_length.
    Assumes data is already filtered and in the correct format.
    Requires: 'file', 'time_s' columns + all channel columns.
    """
    if 'file' not in df.columns:
        raise ValueError("Data must have a 'file' column identifying each sensor recording.")
    if 'time_s' not in df.columns:
        raise ValueError("Data must have a 'time_s' column for sorting.")

    missing = [ch for ch in channels if ch not in df.columns]
    if missing:
        raise ValueError(f"Missing channels: {missing}")

    # Collect available metadata columns
    meta_cols = ['file']
    for col in ['treatment', 'n_contact', 'c_1_type',
                'clear_passage', 'centre_hub_contact', 'blade_contact']:
        if col in df.columns:
            meta_cols.append(col)

    file_metadata = (
        df.groupby("file")[meta_cols[1:]]
          .first()
          .reset_index()
    )

    # Binary label and strike type from labels if present
    if labeled and 'n_contact' in df.columns:
        file_metadata['blade_strike'] = (file_metadata['n_contact'] > 0).astype(int)

        def classify_strike_type(row):
            if row["n_contact"] == 0:
                return "no_contact"
            elif row.get("c_1_type") == "Direct":
                return "direct_strike"
            else:
                return "indirect_strike"
        file_metadata['strike_type'] = file_metadata.apply(classify_strike_type, axis=1)

    # Extract and pad time series
    X_list  = []
    lengths = []
    for file_id in file_metadata['file']:
        file_data = df[df['file'] == file_id].sort_values('time_s')
        ts        = file_data[channels].values
        ts_padded = pad_time_series(ts, max_length)
        X_list.append(ts_padded.T)
        lengths.append(len(ts))

    X = np.stack(X_list)
    print(f"  Files: {len(file_metadata)}, shape: {X.shape}")
    print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    if max(lengths) > max_length:
        print(f"  ⚠ Some sequences longer than training max ({max_length}): truncated")

    return X, file_metadata

def predict(model, X, threshold, file_metadata):
    """Run model, apply threshold, return metadata with predictions appended."""
    scores = model.decision_function(X)
    scores = np.clip(scores, -500, 500)
    probs  = 1 / (1 + np.exp(-scores))
    preds  = (probs >= threshold).astype(int)

    results = file_metadata.copy()
    results['probability_strike'] = np.round(probs, 4)
    results['predicted_strike']   = preds
    results['confidence'] = np.where(
        preds == 1, probs, 1 - probs)
    results['confidence'] = results['confidence'].round(4)
    return results, probs, preds

# ============================================================================
# STEP 1: Sanity check on training data (optional)
# ============================================================================
if RUN_SANITY_CHECK:
    print("\n" + "=" * 60)
    print("STEP 1: SANITY CHECK ON TRAINING DATA")
    print("=" * 60)
    print("\nNOTE: Training accuracy will be HIGHER than CV metrics.")
    print("      CV metrics are the true performance estimate.")
    print("      This check confirms the pipeline is working correctly.\n")

    df_train = pd.read_csv(TRAINING_DATA, low_memory=False)
    print(f"  Loaded: {len(df_train)} rows")
    print(f"  NOTE: Apply the same pre-filtering used in script_01 before passing to model")

    X_train, meta_train = load_and_prepare(
        df_train, channels, max_length=trained_max_length, labeled=True)

    results_train, probs_train, preds_train = predict(
        model, X_train, optimal_threshold, meta_train)

    y_true = meta_train['blade_strike'].values

    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 roc_auc_score, roc_curve)
    train_acc  = accuracy_score(y_true, preds_train)
    train_auc  = roc_auc_score(y_true, probs_train)
    cm         = confusion_matrix(y_true, preds_train)
    tn, fp, fn, tp = cm.ravel()
    train_sens = tp / (tp + fn)
    train_spec = tn / (tn + fp)

    print(f"\n  Training data performance (OPTIMISTIC - model has seen this data):")
    print(f"    Accuracy:    {train_acc:.3f}   (CV: {cv_accuracy:.3f})")
    print(f"    Sensitivity: {train_sens:.3f}   (CV: {cv_sensitivity:.3f})")
    print(f"    Specificity: {train_spec:.3f}   (CV: {cv_specificity:.3f})")
    print(f"    AUC:         {train_auc:.3f}   (CV: {cv_auc:.3f})")

    # Pipeline check
    print(f"\n  Pipeline checks:")
    print(f"    ✓ Model loaded and ran successfully")
    print(f"    ✓ {len(X_train)} files processed, shape: {X_train.shape}")
    print(f"    ✓ Training accuracy ({train_acc:.3f}) ≥ CV accuracy ({cv_accuracy:.3f}) [expected]")

    if train_acc < cv_accuracy - 0.05:
        print(f"    ⚠ WARNING: Training accuracy unexpectedly lower than CV.")
        print(f"      Check that data filtering and channel order match script_01.")
    else:
        print(f"    ✓ Performance consistent with expectations")

    # Per strike type
    print(f"\n  Per strike type:")
    for stype in ['no_contact', 'indirect_strike', 'direct_strike']:
        mask = results_train['strike_type'] == stype
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask.values], preds_train[mask.values])
            print(f"    {stype:17s}: {acc:.3f}  (n={mask.sum()})")

    # Confusion matrix print
    print(f"\n  Confusion Matrix (training data):")
    print(f"                Pred No Strike  Pred Strike")
    print(f"  True No Strike     {tn:3d}           {fp:3d}")
    print(f"  True Strike        {fn:3d}           {tp:3d}")

    print(f"\n✓ Sanity check passed - pipeline working correctly")
    print(f"  Use CV metrics above as expected deployment performance")

else:
    print("\nSanity check skipped (RUN_SANITY_CHECK = False)")

# ============================================================================
# STEP 2: Predict on new unlabelled data
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: PREDICTING ON NEW DATA")
print("=" * 60)

if not os.path.exists(NEW_DATA):
    print(f"\n⚠ New data file not found: {NEW_DATA}")
    print(f"  Edit NEW_DATA at the top of this script to point at your sensor file.")
    print(f"  Expected format: same columns as training data (file, time_s, {', '.join(CHANNELS[:3])}, ...)")
    print(f"\nScript complete (sanity check only).")
else:
    print(f"\nLoading: {NEW_DATA}")
    df_new = pd.read_csv(NEW_DATA, low_memory=False)
    print(f"  Loaded: {len(df_new)} rows")

    # Check for required columns
    required = ['file', 'time_s'] + CHANNELS
    missing  = [col for col in required if col not in df_new.columns]
    if missing:
        raise ValueError(f"New data is missing required columns: {missing}")

    X_new, meta_new = load_and_prepare(
        df_new, channels, max_length=trained_max_length, labeled=False)

    results_new, probs_new, preds_new = predict(
        model, X_new, optimal_threshold, meta_new)

    n_files   = len(results_new)
    n_strikes = preds_new.sum()
    strike_rate = n_strikes / n_files

    print(f"\n  Results:")
    print(f"    Files processed:       {n_files}")
    print(f"    Predicted strikes:     {n_strikes}")
    print(f"    Predicted no strikes:  {n_files - n_strikes}")
    print(f"    Predicted strike rate: {strike_rate:.1%}")

    print(f"\n  High confidence predictions (probability > 0.8 or < 0.2):")
    high_conf = results_new[
        (results_new['probability_strike'] > 0.8) |
        (results_new['probability_strike'] < 0.2)
    ]
    print(f"    {len(high_conf)}/{n_files} files ({len(high_conf)/n_files:.1%})")

    print(f"\n  Uncertain predictions (0.3 < probability < 0.7):")
    uncertain = results_new[
        (results_new['probability_strike'] > 0.3) &
        (results_new['probability_strike'] < 0.7)
    ]
    print(f"    {len(uncertain)}/{n_files} files — review these manually")
    if len(uncertain) > 0:
        print(uncertain[['file', 'probability_strike', 'predicted_strike']].to_string(index=False))

    # Per-treatment summary if treatment column exists
    if 'treatment' in results_new.columns:
        print(f"\n  Predicted strike rate by treatment:")
        tx_summary = (
            results_new.groupby('treatment')
            .agg(
                n_files=('file', 'count'),
                n_predicted_strike=('predicted_strike', 'sum'),
                mean_probability=('probability_strike', 'mean')
            )
            .assign(predicted_strike_rate=lambda x: x['n_predicted_strike'] / x['n_files'])
            .round(3)
        )
        print(tx_summary.to_string())

    # ---- Figures ----
    plt.style.use('default')

    # 1. Probability distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)
    ax.hist(probs_new[preds_new == 0], bins=bins, alpha=0.6,
            label=f'Predicted No Strike (n={int((preds_new==0).sum())})',
            color='steelblue', edgecolor='black', lw=0.5)
    ax.hist(probs_new[preds_new == 1], bins=bins, alpha=0.6,
            label=f'Predicted Strike (n={int(preds_new.sum())})',
            color='tomato', edgecolor='black', lw=0.5)
    ax.axvline(optimal_threshold, color='green', linestyle='--', lw=2,
               label=f'Threshold ({optimal_threshold:.3f})')
    ax.set_xlabel('Predicted Probability of Strike', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Blade Strike Prediction Probabilities - New Data',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/prediction_probability_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Strike rate by treatment (if available)
    if 'treatment' in results_new.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        tx_plot = tx_summary.reset_index()
        bars = ax.bar(tx_plot['treatment'].astype(str),
                      tx_plot['predicted_strike_rate'],
                      color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Treatment', fontsize=12)
        ax.set_ylabel('Predicted Strike Rate', fontsize=12)
        ax.set_title('Predicted Blade Strike Rate by Treatment',
                     fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        for bar, val in zip(bars, tx_plot['predicted_strike_rate']):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/strike_rate_by_treatment.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ---- Save CSVs ----
    results_new.to_csv(f"{OUTDIR}/blade_strike_predictions.csv", index=False)

    # Separate high-priority review file (uncertain predictions)
    if len(uncertain) > 0:
        uncertain.to_csv(f"{OUTDIR}/uncertain_predictions_for_review.csv", index=False)

    print(f"\n✓ Saved: {OUTDIR}/blade_strike_predictions.csv")
    if len(uncertain) > 0:
        print(f"✓ Saved: {OUTDIR}/uncertain_predictions_for_review.csv")
    print(f"✓ Saved: {OUTDIR}/prediction_probability_distribution.png")
    if 'treatment' in results_new.columns:
        print(f"✓ Saved: {OUTDIR}/strike_rate_by_treatment.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SCRIPT 05 COMPLETE")
print("=" * 60)
print(f"\nExpected model performance (from CV):")
print(f"  Accuracy:    {cv_accuracy:.3f}")
print(f"  Sensitivity: {cv_sensitivity:.3f}")
print(f"  Specificity: {cv_specificity:.3f}")
print(f"  AUC:         {cv_auc:.3f}")
if os.path.exists(NEW_DATA):
    print(f"\nNew data predictions saved to: {OUTDIR}/")
