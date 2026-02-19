import pandas as pd
import numpy as np
import json
import os
import joblib
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sktime.transformations.panel.rocket import MiniRocket
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


RAW_DATA    = "python_data/labeled_data_with_types.csv"
BINARY_MODEL = "python_results/binary/final_model_for_deployment.joblib"
METRICS_PATH = "python_results/binary/performance_metrics.json"
OUTDIR      = "python_results/channel_importance"
DATADIR     = "python_data"
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

ALL_CHANNELS = [
    'higacc_x_g', 'higacc_y_g', 'higacc_z_g',
    'inacc_x_ms', 'inacc_y_ms', 'inacc_z_ms',
    'rot_x_degs', 'rot_y_degs', 'rot_z_degs',
    'pressure_kpa'
]

N_PERMUTATIONS = 20

df = pd.read_csv(RAW_DATA, low_memory=False)
# df = df[df["treatment"] != "400 (50%)"].copy()
# df = df[df["centre_hub_contact"] != "CH contact"].copy()
# df = df[df["roi"].str.contains("roi4_nadir", na=False)].copy()

df["blade_strike"] = (df["passage_type"] != "No contact").astype(int)

def classify_strike_type(row):
    if row["passage_type"] == "No contact":
        return "no_contact"
    elif row["passage_type"] == "Leading edge strike":
        return f"leading_{row['leading_type'].lower()}"
    elif row["passage_type"] == "Other impeller collision":
        return f"other_{row['other_type'].lower().replace(' ', '_')}"
    return "unknown"

df["strike_type"] = df.apply(classify_strike_type, axis=1)

file_metadata = (
    df.groupby("file")
      .agg({
          "blade_strike": "first",
          "strike_type":  "first",
          "treatment":    "first"
      })
      .reset_index()
)

print(f"Files: {len(file_metadata)}")
print(f"  Strikes:    {file_metadata['blade_strike'].sum()}")
print(f"  No strikes: {(file_metadata['blade_strike'] == 0).sum()}")

# Extract time series for ALL channels
time_series_dict = {}
lengths = []
for file_id in file_metadata['file']:
    file_data = df[df['file'] == file_id].sort_values('time_s')
    ts_data   = file_data[ALL_CHANNELS].values
    time_series_dict[file_id] = ts_data
    lengths.append(len(ts_data))

max_length = max(lengths)
print(f"Max sequence length: {max_length}")

def pad_time_series(ts, target_length):
    if len(ts) >= target_length:
        return ts[:target_length]
    n_pad = target_length - len(ts)
    return np.vstack([ts, np.tile(ts[-1], (n_pad, 1))])

X_full_list = []
for file_id in file_metadata['file']:
    ts_padded = pad_time_series(time_series_dict[file_id], max_length)
    X_full_list.append(ts_padded.T)

X_full = np.stack(X_full_list)  # (n_samples, 10, n_timepoints)
y      = file_metadata["blade_strike"].values

print(f"Data shapes: X={X_full.shape}, y={y.shape}")

# Class weights for ablation training
class_weights_arr = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights_arr))
sample_weights_full = np.array([class_weight_dict[label] for label in y])

# Channel groups for plotting
def get_channel_group(name):
    if 'higacc'   in name: return 'High Acceleration'
    elif 'inacc'  in name: return 'Internal Acceleration'
    elif 'rot'    in name: return 'Rotation'
    elif 'pressure' in name: return 'Pressure'
    return 'Other'

GROUP_COLORS = {
    'High Acceleration':      '#1f77b4',
    'Internal Acceleration':  '#ff7f0e',
    'Rotation':               '#2ca02c',
    'Pressure':               '#d62728'
}

print(f"\nLoading trained binary model from: {BINARY_MODEL}")

model = joblib.load(BINARY_MODEL)

# Load optimal threshold
with open(METRICS_PATH, 'r') as f:
    binary_metrics = json.load(f)
optimal_threshold = binary_metrics['out_of_fold_performance']['optimal_threshold']
print(f"Optimal threshold: {optimal_threshold:.3f}")

# Baseline performance on training data (for relative comparison)
scores_base = model.decision_function(X_full)
scores_base = np.clip(scores_base, -500, 500)
probs_base  = 1 / (1 + np.exp(-scores_base))
preds_base  = (probs_base >= optimal_threshold).astype(int)
baseline_accuracy = accuracy_score(y, preds_base)

# Baseline indirect accuracy: among indirect strike files, how many correctly predicted as strike
# this needs attention
indirect_mask = (file_metadata["strike_type"] == "leading_indirect")
baseline_indirect_acc = (preds_base[indirect_mask] == y[indirect_mask]).mean() if indirect_mask.sum() > 0 else 0.0

print(f"Baseline accuracy (on training data): {baseline_accuracy:.4f}")
print(f"Baseline indirect accuracy:           {baseline_indirect_acc:.4f}")
print(f"(Note: training accuracy is optimistic; importance is measured relatively)")
print(f"\nShuffling each channel {N_PERMUTATIONS} times...")

perm_results = []
perm_raw = {ch: [] for ch in ALL_CHANNELS}  # store raw drops during first pass

for ch_idx in range(len(ALL_CHANNELS)):
    ch_name = ALL_CHANNELS[ch_idx]
    print(f"\n[{ch_idx+1}/{len(ALL_CHANNELS)}] {ch_name}")

    acc_drops          = []
    indirect_acc_drops = []

    for perm in tqdm(range(N_PERMUTATIONS), desc=f"  Permuting", leave=False):
        try:
            X_shuffled = X_full.copy()
            np.random.shuffle(X_shuffled[:, ch_idx, :])

            scores_shuf = model.decision_function(X_shuffled)
            scores_shuf = np.clip(scores_shuf, -500, 500)
            probs_shuf  = 1 / (1 + np.exp(-scores_shuf))
            preds_shuf  = (probs_shuf >= optimal_threshold).astype(int)

            acc_shuf = accuracy_score(y, preds_shuf)
            acc_drops.append(baseline_accuracy - acc_shuf)
            perm_raw[ch_name].append(baseline_accuracy - acc_shuf)  # store for box plot

            if indirect_mask.sum() > 0:
                indirect_acc_shuf = (preds_shuf[indirect_mask] == y[indirect_mask]).mean()
                indirect_acc_drops.append(baseline_indirect_acc - indirect_acc_shuf)

        except Exception as e:
            print(f"    Warning perm {perm}: {e}")
            continue

    if len(acc_drops) > 0:
        perm_results.append({
            'channel_idx':             ch_idx,
            'channel_name':            ch_name,
            'channel_group':           get_channel_group(ch_name),
            'mean_accuracy_drop':      np.mean(acc_drops),
            'std_accuracy_drop':       np.std(acc_drops),
            'mean_indirect_acc_drop':  np.mean(indirect_acc_drops) if indirect_acc_drops else 0.0,
            'std_indirect_acc_drop':   np.std(indirect_acc_drops)  if indirect_acc_drops else 0.0,
            'n_successful':            len(acc_drops)
        })
        print(f"  Overall drop:  {np.mean(acc_drops):.4f} ± {np.std(acc_drops):.4f}")
        if indirect_acc_drops:
            print(f"  Indirect drop: {np.mean(indirect_acc_drops):.4f} ± {np.std(indirect_acc_drops):.4f}")

perm_df = pd.DataFrame(perm_results).sort_values('mean_accuracy_drop', ascending=False)
perm_df['rank'] = range(1, len(perm_df) + 1)
total_drop = perm_df['mean_accuracy_drop'].sum()
perm_df['pct_contribution'] = (perm_df['mean_accuracy_drop'] / total_drop * 100) if total_drop > 0 else 0

print("\nPermutation")
print("=" * 80)
print(f"{'Rank':<5} {'Channel':<20} {'Acc Drop':<12} {'Std':<10} {'Indirect Drop':<15} {'% Contribution':<15}")
print("-" * 80)
for _, row in perm_df.iterrows():
    print(f"{row['rank']:<5} {row['channel_name']:<20} "
          f"{row['mean_accuracy_drop']:>10.4f}  "
          f"±{row['std_accuracy_drop']:.4f}  "
          f"{row['mean_indirect_acc_drop']:>12.4f}  "
          f"{row['pct_contribution']:>12.1f}%")


configurations = [
    {
        'name':        'higacc_only',
        'description': 'High-acceleration only (3 channels)',
        'channels':    ['higacc_x_g', 'higacc_y_g', 'higacc_z_g']
    },
    {
        'name':        'higacc_plus_pressure',
        'description': 'High-acceleration + Pressure (4 channels)',
        'channels':    ['higacc_x_g', 'higacc_y_g', 'higacc_z_g', 'pressure_kpa']
    },
    {
        'name':        'full_suite',
        'description': 'Full sensor suite (10 channels)',
        'channels':    ALL_CHANNELS
    }
]

print(f"\nConfigurations:")
for cfg in configurations:
    print(f"  {cfg['name']:25s}: {len(cfg['channels'])} channels - {cfg['description']}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ablation_results = []

total_start = time.time()

for cfg_idx, cfg in enumerate(configurations):
    print(f"\n{'='*70}")
    print(f"Config {cfg_idx+1}/{len(configurations)}: {cfg['name']}")
    print(f"{'='*70}")
    print(f"Channels: {cfg['channels']}")

    ch_indices = [ALL_CHANNELS.index(ch) for ch in cfg['channels']]
    X_cfg      = X_full[:, ch_indices, :]

    oof_probs_cfg = np.zeros(len(y))
    oof_preds_cfg = np.zeros(len(y), dtype=int)
    fold_aucs = []

    cfg_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_cfg, y)):
        pipeline = make_pipeline(
            MiniRocket(random_state=42, n_jobs=-1),
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        )

        fold_weights = sample_weights_full[train_idx]
        pipeline.fit(X_cfg[train_idx], y[train_idx],
                     ridgeclassifiercv__sample_weight=fold_weights)

        scores = pipeline.decision_function(X_cfg[test_idx])
        probs  = 1 / (1 + np.exp(-scores))

        # Youden's threshold
        fpr_f, tpr_f, thresh_f = roc_curve(y[test_idx], probs)
        opt_thresh = thresh_f[np.argmax(tpr_f - fpr_f)]

        oof_probs_cfg[test_idx] = probs
        oof_preds_cfg[test_idx] = (probs >= opt_thresh).astype(int)
        fold_aucs.append(roc_auc_score(y[test_idx], probs))

    # Overall metrics
    final_auc_cfg = roc_auc_score(y, oof_probs_cfg)
    fpr_a, tpr_a, thresh_a = roc_curve(y, oof_probs_cfg)
    opt_thresh_overall = thresh_a[np.argmax(tpr_a - fpr_a)]
    final_preds = (oof_probs_cfg >= opt_thresh_overall).astype(int)

    cv_accuracy = accuracy_score(y, final_preds)
    cm_cfg      = confusion_matrix(y, final_preds)
    tn_c, fp_c, fn_c, tp_c = cm_cfg.ravel()

    sens = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
    spec = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0
    prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
    f1   = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0

    # Per-strike-type accuracy
    file_metadata_temp = file_metadata.copy()
    file_metadata_temp['oof_pred'] = final_preds
    direct_acc   = (file_metadata_temp[file_metadata_temp['strike_type']=='leading_direct']['blade_strike'] ==
                file_metadata_temp[file_metadata_temp['strike_type']=='leading_direct']['oof_pred']).mean()
    indirect_acc = (file_metadata_temp[file_metadata_temp['strike_type']=='leading_indirect']['blade_strike'] ==
                file_metadata_temp[file_metadata_temp['strike_type']=='leading_indirect']['oof_pred']).mean()

    cfg_time = time.time() - cfg_start

    ablation_results.append({
        'configuration':    cfg['name'],
        'description':      cfg['description'],
        'n_channels':       len(cfg['channels']),
        'cv_accuracy':      cv_accuracy,
        'cv_auc':           final_auc_cfg,
        'cv_auc_std':       np.std(fold_aucs),
        'sensitivity':      sens,
        'specificity':      spec,
        'precision':        prec,
        'f1_score':         f1,
        'direct_accuracy':  direct_acc,
        'indirect_accuracy': indirect_acc,
        'confusion_matrix': cm_cfg.tolist(),
        'training_time_sec': cfg_time
    })

    print(f"\n  CV Accuracy:       {cv_accuracy:.4f}")
    print(f"  CV AUC:            {final_auc_cfg:.4f} (±{np.std(fold_aucs):.4f})")
    print(f"  Sensitivity:       {sens:.4f}")
    print(f"  Specificity:       {spec:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  Direct accuracy:   {direct_acc:.4f}")
    print(f"  Indirect accuracy: {indirect_acc:.4f}")
    print(f"  Time: {cfg_time:.1f}s")

print(f"\nTotal ablation time: {(time.time()-total_start)/60:.1f} minutes")

ablation_df = pd.DataFrame(ablation_results)

# Calculate improvements vs higacc_only baseline
baseline_row    = ablation_df[ablation_df['configuration'] == 'higacc_only'].iloc[0]
baseline_acc    = baseline_row['cv_accuracy']
baseline_indirect = baseline_row['indirect_accuracy']

ablation_df['accuracy_gain_vs_minimal']  = ablation_df['cv_accuracy']        - baseline_acc
ablation_df['indirect_gain_vs_minimal']  = ablation_df['indirect_accuracy']   - baseline_indirect

print(f"{'Config':<25} {'Ch':>3} {'Accuracy':>10} {'AUC':>8} {'Sensitivity':>12} "
      f"{'Indirect':>10} {'Gain':>8}")

for _, row in ablation_df.iterrows():
    gain = f"+{row['accuracy_gain_vs_minimal']*100:.1f}%" if row['accuracy_gain_vs_minimal'] > 0 else f"{row['accuracy_gain_vs_minimal']*100:.1f}%"
    print(f"{row['description']:<25} {row['n_channels']:>3} "
          f"{row['cv_accuracy']:>10.4f} {row['cv_auc']:>8.4f} "
          f"{row['sensitivity']:>12.4f} {row['indirect_accuracy']:>10.4f} {gain:>8}")

plt.style.use('default')
sns.set_palette("husl")


# 2. Channel importance box plot
fig, ax = plt.subplots(figsize=(10, 7))
ordered_channels = perm_df['channel_name'].tolist()
box_data   = [perm_raw[ch] for ch in ordered_channels]
box_colors = [GROUP_COLORS[get_channel_group(ch)] for ch in ordered_channels]
bp = ax.boxplot(box_data, labels=ordered_channels, vert=False,
                patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xlabel('Accuracy Drop', fontsize=12, fontweight='bold')
ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
ax.set_title(f'Distribution of Permutation Importance\n({N_PERMUTATIONS} permutations per channel)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/channel_importance_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()
# 3. Permutation heatmap
perm_matrix = np.array([[perm_raw[ch][p] if p < len(perm_raw[ch]) else np.nan
                          for p in range(N_PERMUTATIONS)]
                         for ch in ordered_channels])
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(perm_matrix, ax=ax, cmap='YlOrRd',
            yticklabels=ordered_channels,
            xticklabels=[str(i+1) for i in range(N_PERMUTATIONS)],
            cbar_kws={'label': 'Accuracy Drop'},
            linewidths=0.3, linecolor='gray')
ax.set_xlabel('Permutation Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
ax.set_title('Permutation Importance Heatmap (Accuracy Drop)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/permutation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Permutation scatter: overall vs indirect accuracy drop
fig, ax = plt.subplots(figsize=(9, 7))
bar_colors_s = [GROUP_COLORS[get_channel_group(ch)] for ch in perm_df['channel_name']]
ax.scatter(perm_df['mean_accuracy_drop'], perm_df['mean_indirect_acc_drop'],
           s=100, c=bar_colors_s, edgecolors='black', zorder=5, alpha=0.8)
for _, row in perm_df.iterrows():
    ax.annotate(row['channel_name'],
                (row['mean_accuracy_drop'], row['mean_indirect_acc_drop']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.axvline(0, color='black', lw=0.8, linestyle='--')
ax.set_xlabel('Overall Accuracy Drop', fontsize=12, fontweight='bold')
ax.set_ylabel('Indirect Strike Accuracy Drop', fontsize=12, fontweight='bold')
ax.set_title('Permutation Importance: Overall vs Indirect\n(Binary Model)',
             fontsize=13, fontweight='bold')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=g, alpha=0.8)
                   for g in GROUP_COLORS if g in perm_df['channel_group'].values]
ax.legend(handles=legend_elements, fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/permutation_importance_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

config_labels  = ['Minimal\n(3 ch)', 'Medium\n(4 ch)', 'Full\n(10 ch)']
config_colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']


fig, axes = plt.subplots(1, 3, figsize=(14, 6))
metrics_to_plot = [
    ('cv_accuracy',      'Overall Accuracy'),
    ('cv_auc',           'ROC-AUC'),
    ('indirect_accuracy','Indirect Strike Accuracy')
]
for ax, (metric, title) in zip(axes, metrics_to_plot):
    values = ablation_df[metric].values
    bars   = ax.bar(config_labels, values, color=config_colors, alpha=0.8, edgecolor='black')
    ax.set_ylim([max(0, min(values)*0.95), 1.0])
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.suptitle('Sensor Configuration Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/configuration_comparison.png", dpi=300, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(9, 7))
bar_colors_s = [GROUP_COLORS[get_channel_group(ch)] for ch in perm_df['channel_name']]
ax.scatter(perm_df['mean_accuracy_drop'], perm_df['mean_indirect_acc_drop'],
           s=100, c=bar_colors_s, edgecolors='black', zorder=5, alpha=0.8)
for _, row in perm_df.iterrows():
    ax.annotate(row['channel_name'],
                (row['mean_accuracy_drop'], row['mean_indirect_acc_drop']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.axvline(0, color='black', lw=0.8, linestyle='--')
ax.set_xlabel('Overall Accuracy Drop', fontsize=12, fontweight='bold')
ax.set_ylabel('Indirect Strike Accuracy Drop', fontsize=12, fontweight='bold')
ax.set_title('Permutation Importance: Overall vs Indirect\n(Binary Model)',
             fontsize=13, fontweight='bold')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=g, alpha=0.8)
                   for g in GROUP_COLORS if g in perm_df['channel_group'].values]
ax.legend(handles=legend_elements, fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/importance_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Ablation heatmap
fig, ax = plt.subplots(figsize=(10, 5))
heatmap_data = ablation_df[['cv_accuracy', 'cv_auc', 'sensitivity', 'specificity',
                             'indirect_accuracy']].set_index(ablation_df['description'])
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Score'})
ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/ablation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Accuracy drop chart (full 10ch vs each ablation - indirect focus)
full_acc      = ablation_df[ablation_df['configuration'] == 'full_suite']['cv_accuracy'].values[0]
full_indirect = ablation_df[ablation_df['configuration'] == 'full_suite']['indirect_accuracy'].values[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
drop_overall = perm_df['mean_accuracy_drop'].values
bar_cols = [GROUP_COLORS[g] for g in perm_df['channel_group'].values]
bars = ax1.bar(perm_df['channel_name'], drop_overall, color=bar_cols, alpha=0.8, edgecolor='black')
ax1.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='1% threshold')
ax1.set_xlabel('Channel Removed', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy Drop', fontsize=11, fontweight='bold')
ax1.set_title('Overall Accuracy Drop\n(Permutation)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

ax2 = axes[1]
acc_values = ablation_df['cv_accuracy'].values
ax2.bar(config_labels, acc_values, color=config_colors, alpha=0.8, edgecolor='black')
for i, (bar, val) in enumerate(zip(ax2.patches, acc_values)):
    gain = val - baseline_acc
    label = f'{val:.3f}\n(base)' if i == 0 else f'{val:.3f}\n({gain:+.3f})'
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             label, ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylabel('CV Accuracy', fontsize=11, fontweight='bold')
ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
ax2.set_title('Accuracy by Sensor Configuration\n(Ablation)', fontsize=12, fontweight='bold')
ax2.set_ylim([max(0, baseline_acc * 0.9), 1.0])
ax2.grid(alpha=0.3, axis='y')

plt.suptitle('Channel Importance: Permutation vs Ablation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/indirect_drop_by_channel.png", dpi=300, bbox_inches='tight')
plt.close()

# Permutation importance ranking (sorted)
perm_df.to_csv(f"{OUTDIR}/channel_importance_ranking.csv", index=False)

# Detailed permutation results (raw drops per permutation)
perm_detailed_rows = []
for ch_name, drops in perm_raw.items():
    for perm_n, drop_val in enumerate(drops):
        perm_detailed_rows.append({
            'channel_name': ch_name,
            'channel_group': get_channel_group(ch_name),
            'permutation': perm_n,
            'accuracy_drop': drop_val
        })
perm_detailed_df = pd.DataFrame(perm_detailed_rows)
perm_detailed_df.to_csv(f"{OUTDIR}/permutation_importance_detailed.csv", index=False)

# Ablation detailed
ablation_df.to_csv(f"{OUTDIR}/ablation_study_detailed.csv", index=False)

# Sensor comparison results (clean version)
sensor_comparison = ablation_df[['configuration', 'description', 'n_channels',
                                  'cv_accuracy', 'cv_auc', 'sensitivity', 'specificity',
                                  'precision', 'f1_score', 'direct_accuracy',
                                  'indirect_accuracy']].copy()
sensor_comparison.to_csv(f"{OUTDIR}/sensor_comparison_results.csv", index=False)

# Model comparison table (combining key metrics)
model_comparison = pd.DataFrame([
    {
        'model':         'Binary (full suite)',
        'configuration': 'full_suite',
        'n_channels':    10,
        'cv_accuracy':   ablation_df[ablation_df['configuration']=='full_suite']['cv_accuracy'].values[0],
        'cv_auc':        ablation_df[ablation_df['configuration']=='full_suite']['cv_auc'].values[0],
        'indirect_accuracy': ablation_df[ablation_df['configuration']=='full_suite']['indirect_accuracy'].values[0]
    },
    {
        'model':         'Binary (higacc + pressure)',
        'configuration': 'higacc_plus_pressure',
        'n_channels':    4,
        'cv_accuracy':   ablation_df[ablation_df['configuration']=='higacc_plus_pressure']['cv_accuracy'].values[0],
        'cv_auc':        ablation_df[ablation_df['configuration']=='higacc_plus_pressure']['cv_auc'].values[0],
        'indirect_accuracy': ablation_df[ablation_df['configuration']=='higacc_plus_pressure']['indirect_accuracy'].values[0]
    },
    {
        'model':         'Binary (higacc only)',
        'configuration': 'higacc_only',
        'n_channels':    3,
        'cv_accuracy':   ablation_df[ablation_df['configuration']=='higacc_only']['cv_accuracy'].values[0],
        'cv_auc':        ablation_df[ablation_df['configuration']=='higacc_only']['cv_auc'].values[0],
        'indirect_accuracy': ablation_df[ablation_df['configuration']=='higacc_only']['indirect_accuracy'].values[0]
    }
])
model_comparison.to_csv(f"{OUTDIR}/model_comparison.csv", index=False)


# Save JSON summary
full_row    = ablation_df[ablation_df['configuration']=='full_suite'].iloc[0]
minimal_row = ablation_df[ablation_df['configuration']=='higacc_only'].iloc[0]

summary = {
    "permutation_importance": {
        "n_permutations":      N_PERMUTATIONS,
        "baseline_accuracy":   float(baseline_accuracy),
        "most_important":      perm_df.iloc[0]['channel_name'],
        "least_important":     perm_df.iloc[-1]['channel_name'],
        "top_3_channels":      perm_df.head(3)['channel_name'].tolist()
    },
    "ablation_study": {
        "configurations_tested": 3,
        "higacc_only_accuracy":        float(minimal_row['cv_accuracy']),
        "higacc_plus_pressure_accuracy": float(ablation_df[ablation_df['configuration']=='higacc_plus_pressure']['cv_accuracy'].values[0]),
        "full_suite_accuracy":          float(full_row['cv_accuracy']),
        "accuracy_gain_minimal_to_full": float(full_row['cv_accuracy'] - minimal_row['cv_accuracy']),
        "publication_claim": (
            f"High-acceleration sensors alone achieve {minimal_row['cv_accuracy']:.1%} accuracy. "
            f"Adding all remaining sensors improves performance by "
            f"{(full_row['cv_accuracy'] - minimal_row['cv_accuracy'])*100:.1f} percentage points."
        )
    }
}

with open(f"{OUTDIR}/channel_importance_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Most important channel: {perm_df.iloc[0]['channel_name']} "
      f"(drop: {perm_df.iloc[0]['mean_accuracy_drop']:.4f})")

print(f"\nOutputs: {OUTDIR}/")
