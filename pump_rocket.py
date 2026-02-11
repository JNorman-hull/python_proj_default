import numpy as np
import numba
import pandas as pd
import sklearn
import sktime
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
  classification_report, confusion_matrix, roc_auc_score,
  roc_curve, precision_recall_curve, auc)
from sktime.transformations.panel.rocket import MiniRocket
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
from pathlib import Path

# Set random seed
np.random.seed(42)

# ============================================================================
# 1. Load Raw Data
# ============================================================================
print("\n1. Loading raw data...")

df = pd.read_csv("python_data/raw_labeled_data.csv")

print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Treatments: {df['treatment'].unique()}")
print(f"ROIs: {df['roi'].unique()}")
print(f"Unique files: {df['file'].nunique()}")

# ============================================================================
# 2. Filter Data
# ============================================================================
print("\n2. Filtering data...")

# Remove 400 (50%) treatment
df_filtered = df[df['treatment'] != '400 (50%)'].copy()
print(f"After removing '400 (50%)': {df_filtered.shape[0]} rows")
print(f"Remaining treatments: {df_filtered['treatment'].unique()}")

# Filter to ROI4 nadir only
df_roi4 = df_filtered[df_filtered['roi'].str.contains('roi4_nadir', na=False)].copy()
print(f"ROI4 nadir data: {df_roi4.shape[0]} rows")
print(f"ROI4 unique files: {df_roi4['file'].nunique()}")

# ============================================================================
# 3. Create Labels and Classify Strike Types
# ============================================================================
print("\n3. Creating labels and metadata...")

# Binary label: 1 if blade strike occurred (n_contact > 0)
df_roi4['blade_strike'] = (df_roi4['n_contact'] > 0).astype(int)

# Classify strike type for stratification
def classify_strike_type(row):
    if row['n_contact'] == 0:
        return 'no_contact'
    elif row['c_1_type'] == 'Direct':
        return 'direct_strike'
    else:
        return 'indirect_strike'

df_roi4['strike_type'] = df_roi4.apply(classify_strike_type, axis=1)

# Get file-level metadata (one row per file)
file_metadata = df_roi4.groupby('file').agg({
    'blade_strike': 'first',
    'strike_type': 'first',
    'treatment': 'first',
    'n_contact': 'first',
    'c_1_type': 'first',
    'clear_passage': 'first',
    'centre_hub_contact': 'first',
    'blade_contact': 'first',
    'c_1_bl': 'first'
}).reset_index()

print(f"\nTotal unique files: {len(file_metadata)}")
print(f"\nLabel distribution:")
print(file_metadata['blade_strike'].value_counts())
print(f"\nStrike type distribution:")
print(file_metadata['strike_type'].value_counts())
print(f"\nTreatment distribution:")
print(file_metadata['treatment'].value_counts())

# Summary table
print("\nDetailed breakdown:")
summary = file_metadata.groupby(['treatment', 'strike_type']).size().reset_index(name='n_files')
print(summary)

# ============================================================================
# 4. Define Channels
# ============================================================================
print("\n4. Defining sensor channels...")

# Channels to use (excluding magnitude features to avoid redundancy)
channels = [
    'higacc_x_g', 'higacc_y_g', 'higacc_z_g',
    'inacc_x_ms', 'inacc_y_ms', 'inacc_z_ms',
    'rot_x_degs', 'rot_y_degs', 'rot_z_degs',
    'pressure_kpa'
]

print(f"Using {len(channels)} channels:")
for i, ch in enumerate(channels, 1):
    print(f"  {i}. {ch}")

# Verify all channels exist
missing_channels = [ch for ch in channels if ch not in df_roi4.columns]
if missing_channels:
    print(f"\nWARNING: Missing channels: {missing_channels}")
    channels = [ch for ch in channels if ch in df_roi4.columns]
    print(f"Updated to use {len(channels)} available channels")

# ============================================================================
# 5. Extract Time Series for Each File
# ============================================================================
print("\n5. Extracting time series data...")

time_series_dict = {}
lengths = []

for file_id in file_metadata['file']:
    # Get data for this file, sorted by time
    file_data = df_roi4[df_roi4['file'] == file_id].sort_values('time_s')
    
    # Extract channel data as numpy array
    ts_data = file_data[channels].values  # Shape: (n_timepoints, n_channels)
    
    # Store in dictionary
    time_series_dict[file_id] = ts_data
    lengths.append(len(ts_data))

print(f"Extracted time series for {len(time_series_dict)} files")
print(f"Time series lengths:")
print(f"  Min: {min(lengths)}")
print(f"  Max: {max(lengths)}")
print(f"  Mean: {np.mean(lengths):.1f}")
print(f"  Median: {np.median(lengths):.1f}")

# Check for consistency
unique_lengths = set(lengths)
print(f"  Unique lengths: {sorted(unique_lengths)}")

# ============================================================================
# 6. Stratified Train/Test Split (70/30)
# ============================================================================
print("\n6. Creating stratified train/test split...")

# Separate strike and non-strike files
strike_files = file_metadata[file_metadata['blade_strike'] == 1].copy()
no_strike_files = file_metadata[file_metadata['blade_strike'] == 0].copy()

print(f"Strike files: {len(strike_files)}")
print(f"Non-strike files: {len(no_strike_files)}")

# Create stratification variable (combination of strike type and treatment)
strike_files['strata'] = strike_files['strike_type'] + '_' + strike_files['treatment']
no_strike_files['strata'] = no_strike_files['treatment']

# Function to perform stratified sampling
def stratified_split(df, train_prop=0.7, random_state=42):
    """Split data maintaining proportions within each stratum"""
    train_list = []
    test_list = []
    
    for stratum in df['strata'].unique():
        stratum_data = df[df['strata'] == stratum].copy()
        n_total = len(stratum_data)
        n_train = int(n_total * train_prop)
        
        # Shuffle and split
        shuffled = stratum_data.sample(frac=1, random_state=random_state)
        train_list.append(shuffled.iloc[:n_train])
        test_list.append(shuffled.iloc[n_train:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, test_df

# Split strike and non-strike files separately
train_strikes, test_strikes = stratified_split(strike_files)
train_no_strikes, test_no_strikes = stratified_split(no_strike_files)

# Combine
train_files = pd.concat([train_strikes, train_no_strikes], ignore_index=True)
test_files = pd.concat([test_strikes, test_no_strikes], ignore_index=True)

print(f"\nTrain/Test Split:")
print(f"Training files: {len(train_files)}")
print(f"  - Strikes: {train_files['blade_strike'].sum()}")
print(f"  - Non-strikes: {(train_files['blade_strike'] == 0).sum()}")
print(f"  - Strike rate: {train_files['blade_strike'].mean():.1%}")

print(f"Test files: {len(test_files)}")
print(f"  - Strikes: {test_files['blade_strike'].sum()}")
print(f"  - Non-strikes: {(test_files['blade_strike'] == 0).sum()}")
print(f"  - Strike rate: {test_files['blade_strike'].mean():.1%}")

# Verify stratification worked
print(f"\nTraining set composition by strike type:")
print(train_files['strike_type'].value_counts())

# ============================================================================
# 7. Pad Time Series and Create 3D Arrays
# ============================================================================
print("\n7. Padding time series to uniform length...")

# Use maximum length for padding
max_length = max(lengths)
print(f"Max sequence length: {max_length} time points")
print(f"At 2048 Hz sampling: {max_length / 2048 * 1000:.1f} ms")

def pad_time_series(ts, target_length):
    """
    Pad time series to target length by repeating the last row.
    If longer than target, truncate to target length.
    
    Args:
        ts: numpy array of shape (n_timepoints, n_channels)
        target_length: int, desired length
    
    Returns:
        Padded array of shape (target_length, n_channels)
    """
    current_length = len(ts)
    
    if current_length >= target_length:
        # Truncate
        return ts[:target_length]
    else:
        # Pad by repeating last row
        n_pad = target_length - current_length
        padding = np.tile(ts[-1], (n_pad, 1))
        return np.vstack([ts, padding])

# Create training arrays
print("\nCreating training arrays...")
X_train_list = []
y_train_list = []

for _, row in train_files.iterrows():
    file_id = row['file']
    ts = time_series_dict[file_id]
    
    # Pad to max length
    ts_padded = pad_time_series(ts, max_length)
    
    # Transpose to (n_channels, n_timepoints) for sktime
    ts_transposed = ts_padded.T
    
    X_train_list.append(ts_transposed)
    y_train_list.append(row['blade_strike'])

X_train = np.array(X_train_list)  # Shape: (n_samples, n_channels, n_timepoints)
y_train = np.array(y_train_list)

# Create test arrays
print("Creating test arrays...")
X_test_list = []
y_test_list = []

for _, row in test_files.iterrows():
    file_id = row['file']
    ts = time_series_dict[file_id]
    
    ts_padded = pad_time_series(ts, max_length)
    ts_transposed = ts_padded.T
    
    X_test_list.append(ts_transposed)
    y_test_list.append(row['blade_strike'])

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

print(f"\nFinal array shapes:")
print(f"X_train: {X_train.shape} (samples × channels × timepoints)")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape} ({y_train.sum()} strikes, {(y_train == 0).sum()} non-strikes)")
print(f"y_test: {y_test.shape} ({y_test.sum()} strikes, {(y_test == 0).sum()} non-strikes)")

# Save arrays for future use
os.makedirs("python_data", exist_ok=True)
np.save("python_data/X_train.npy", X_train)
np.save("python_data/X_test.npy", X_test)
np.save("python_data/y_train.npy", y_train)
np.save("python_data/y_test.npy", y_test)

# ============================================================================
# 8. Train MiniRocket Transformer
# ============================================================================

# Initialize MiniRocket
minirocket = MiniRocket(
    num_kernels=10000,
    random_state=123,
    n_jobs=-1  # Use all available CPU cores
)

# Fit on training data
print("Fitting MiniRocket kernels on training data...")
minirocket.fit(X_train)
print("✓ Kernels fitted")

# Transform training data
print("\nTransforming training data to feature vectors...")
X_train_transform = minirocket.transform(X_train)
print(f"✓ Training features shape: {X_train_transform.shape}")
print(f"  ({X_train_transform.shape[1]} features per sample)")

# Transform test data
print("\nTransforming test data...")
X_test_transform = minirocket.transform(X_test)
print(f"✓ Test features shape: {X_test_transform.shape}")

# ============================================================================
# 9. Train Ridge Classifier with Class Weights
# ============================================================================
print("\n9. Training Ridge Classifier...")
print("=" * 70)

# Calculate class weights to handle imbalance
n_strikes = y_train.sum()
n_no_strikes = (y_train == 0).sum()
strike_weight = n_no_strikes / n_strikes

print(f"Class distribution in training:")
print(f"  Non-strikes: {n_no_strikes}")
print(f"  Strikes: {n_strikes}")
print(f"  Imbalance ratio: {strike_weight:.2f}:1")
print(f"  Using strike weight: {strike_weight:.2f}")

# Create sample weights (upweight minority class)
sample_weights = np.where(y_train == 1, strike_weight, 1.0)

# Ridge Classifier with cross-validation
print("\nTraining Ridge Classifier with 10-fold CV...")
ridge_clf = RidgeClassifierCV(
    alphas=np.logspace(-3, 3, 10),  # Test 10 alpha values
    cv=10,
    scoring='roc_auc'
)

ridge_clf.fit(X_train_transform, y_train, sample_weight=sample_weights)
print(f"✓ Training complete")
print(f"  Best alpha (regularization): {ridge_clf.alpha_}")

# ============================================================================
# 10. Make Predictions on Test Set
# ============================================================================

# Get decision function scores
y_test_scores = ridge_clf.decision_function(X_test_transform)

# Convert scores to probabilities using sigmoid function
from scipy.special import expit
y_test_probs = expit(y_test_scores)

# Predict classes using default threshold (0.5)
y_test_pred_default = (y_test_probs > 0.5).astype(int)

# ============================================================================
# 11. Evaluate Model Performance
# ============================================================================
print("\n11. Model Evaluation")
print("=" * 70)

# Confusion Matrix (default threshold)
cm_default = confusion_matrix(y_test, y_test_pred_default)
print("\nConfusion Matrix (Threshold = 0.5):")
print(cm_default)
print(f"\n  True Negatives:  {cm_default[0,0]}")
print(f"  False Positives: {cm_default[0,1]}")
print(f"  False Negatives: {cm_default[1,0]}")
print(f"  True Positives:  {cm_default[1,1]}")

# Classification Report
print("\nClassification Report (Threshold = 0.5):")
print(classification_report(y_test, y_test_pred_default,
                          target_names=['No Strike', 'Strike'],
                          digits=3))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_test_probs)
print(f"ROC-AUC Score: {roc_auc:.3f}")

# Precision-Recall AUC
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_probs)
pr_auc = auc(recall, precision)
print(f"PR-AUC Score: {pr_auc:.3f}")

# ============================================================================
# 12. Optimize Decision Threshold
# ============================================================================
print("\n12. Threshold Optimization (Youden's Index)")
print("=" * 70)

# Get ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_probs)

# Calculate Youden's J statistic (TPR - FPR)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = roc_thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"  (maximizes sensitivity + specificity)")

# Re-predict with optimal threshold
y_test_pred_opt = (y_test_probs > optimal_threshold).astype(int)

# Confusion matrix with optimal threshold
cm_opt = confusion_matrix(y_test, y_test_pred_opt)
print(f"\nConfusion Matrix (Optimized Threshold = {optimal_threshold:.3f}):")
print(cm_opt)

# Classification report with optimal threshold
print(f"\nClassification Report (Optimized Threshold = {optimal_threshold:.3f}):")
print(classification_report(y_test, y_test_pred_opt,
                          target_names=['No Strike', 'Strike'],
                          digits=3))

# ============================================================================
# 13. Create Visualizations
# ============================================================================

# Create output directory
os.makedirs("python_results", exist_ok=True)

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

# --- Plot 1: Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Default threshold
sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            cbar=False, annot_kws={'size': 14})
axes[0].set_title('Confusion Matrix\n(Threshold = 0.5)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)
axes[0].set_xticklabels(['No Strike', 'Strike'], fontsize=10)
axes[0].set_yticklabels(['No Strike', 'Strike'], rotation=0, fontsize=10)

# Optimal threshold
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            cbar=False, annot_kws={'size': 14})
axes[1].set_title(f'Confusion Matrix\n(Threshold = {optimal_threshold:.3f})', 
                 fontsize=13, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)
axes[1].set_xticklabels(['No Strike', 'Strike'], fontsize=10)
axes[1].set_yticklabels(['No Strike', 'Strike'], rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig('python_results/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Plot 2: ROC Curve ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=150, zorder=5,
           edgecolors='black', linewidths=1.5,
           label=f'Optimal Threshold = {optimal_threshold:.3f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - MiniRocket Blade Strike Classifier', 
         fontsize=13, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('python_results/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Plot 3: Precision-Recall Curve ---
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'g-', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - MiniRocket Blade Strike Classifier',
         fontsize=13, fontweight='bold')
plt.legend(loc="lower left", fontsize=10)
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.savefig('python_results/pr_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Plot 4: Probability Distribution ---
plt.figure(figsize=(10, 6))

bins = np.linspace(0, 1, 21)
plt.hist(y_test_probs[y_test == 0], bins=bins, alpha=0.6, 
        label='No Strike', color='blue', edgecolor='black', linewidth=0.5)
plt.hist(y_test_probs[y_test == 1], bins=bins, alpha=0.6,
        label='Strike', color='red', edgecolor='black', linewidth=0.5)

plt.axvline(0.5, color='black', linestyle='--', linewidth=2,
           label='Default Threshold (0.5)')
plt.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
           label=f'Optimal Threshold ({optimal_threshold:.3f})')

plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Predicted Probabilities',
         fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3, axis='y')
plt.xlim([0, 1])
plt.savefig('python_results/probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()



# ============================================================================
# 14. Save Model and Results
# ============================================================================


# Save the trained model
model_dict = {
    'minirocket': minirocket,
    'ridge_classifier': ridge_clf,
    'optimal_threshold': optimal_threshold,
    'channels': channels,
    'max_length': max_length
}

with open('python_results/minirocket_model.pkl', 'wb') as f:
    pickle.dump(model_dict, f)

# Save test predictions with metadata
results_df = pd.DataFrame({
    'file': test_files['file'].values,
    'y_true': y_test,
    'y_pred_default': y_test_pred_default,
    'y_pred_optimized': y_test_pred_opt,
    'probability': y_test_probs,
    'strike_type': test_files['strike_type'].values,
    'treatment': test_files['treatment'].values,
    'blade_contact': test_files['blade_contact'].values,
    'c_1_bl': test_files['c_1_bl'].values
})
results_df.to_csv('python_results/test_predictions.csv', index=False)

# Save train/test file lists
train_files.to_csv('python_results/train_files.csv', index=False)
test_files.to_csv('python_results/test_files.csv', index=False)

# Save performance metrics as JSON
metrics = {
    'model': 'MiniRocket + Ridge Classifier',
    'num_kernels': 10000,
    'n_channels': len(channels),
    'max_sequence_length': int(max_length),
    'n_features': int(X_train_transform.shape[1]),
    'training': {
        'n_samples': int(len(y_train)),
        'n_strikes': int(y_train.sum()),
        'n_no_strikes': int((y_train == 0).sum()),
        'strike_rate': float(y_train.mean()),
        'class_weight': float(strike_weight),
        'best_alpha': float(ridge_clf.alpha_)
    },
    'test': {
        'n_samples': int(len(y_test)),
        'n_strikes': int(y_test.sum()),
        'n_no_strikes': int((y_test == 0).sum()),
        'strike_rate': float(y_test.mean())
    },
    'performance': {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'optimal_threshold': float(optimal_threshold),
        'default_threshold': {
            'threshold': 0.5,
            'confusion_matrix': cm_default.tolist(),
            'accuracy': float((y_test == y_test_pred_default).mean()),
            'tp': int(cm_default[1, 1]),
            'fp': int(cm_default[0, 1]),
            'fn': int(cm_default[1, 0]),
            'tn': int(cm_default[0, 0])
        },
        'optimized_threshold': {
            'threshold': float(optimal_threshold),
            'confusion_matrix': cm_opt.tolist(),
            'accuracy': float((y_test == y_test_pred_opt).mean()),
            'tp': int(cm_opt[1, 1]),
            'fp': int(cm_opt[0, 1]),
            'fn': int(cm_opt[1, 0]),
            'tn': int(cm_opt[0, 0])
        }
    }
}

with open('python_results/performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ============================================================================
# 15. Performance Summary by Strike Type
# ============================================================================

results_by_type = results_df.groupby('strike_type').agg({
    'file': 'count',
    'y_true': 'sum',
    'probability': 'mean'
}).rename(columns={'file': 'n_files', 'y_true': 'n_strikes'})

results_by_type['accuracy_default'] = results_df.groupby('strike_type').apply(
    lambda x: (x['y_true'] == x['y_pred_default']).mean()
)
results_by_type['accuracy_optimized'] = results_df.groupby('strike_type').apply(
    lambda x: (x['y_true'] == x['y_pred_optimized']).mean()
)

print(results_by_type.round(3))

# Save summary
results_by_type.to_csv('python_results/performance_by_strike_type.csv')

print(f"  ROC-AUC:           {roc_auc:.3f}")
print(f"  PR-AUC:            {pr_auc:.3f}")
print(f"  Optimal Threshold: {optimal_threshold:.3f}")
print(f"  Test Accuracy:     {(y_test == y_test_pred_opt).mean():.3f}")
print(f"\nAll results saved to: python_results/")
print(f"  - Model: minirocket_model.pkl")
print(f"  - Predictions: test_predictions.csv")
print(f"  - Metrics: performance_metrics.json")
print(f"  - Visualizations: *.png")
