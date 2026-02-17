# ============================================================================
# script_04_channel_importance_multiclass.py
# Channel importance analysis for MULTI-CLASS blade strike classifier
# Mirror of script_03 - use if interested in multiclass channel behaviour
#
# PART A: Permutation importance (accuracy-based) on 10-channel multiclass model
# PART B: Channel ablation study - 3 sensor configurations (multiclass labels)
#         - Config 1: High-acceleration only      (3 channels)
#         - Config 2: High-acceleration + pressure (4 channels)
#         - Config 3: Full sensor suite            (10 channels)
# ============================================================================

import pandas as pd
import numpy as np
import json
import os
import joblib
import time
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
from tqdm import tqdm

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RAW_DATA         = "python_data/raw_labeled_data2.csv"
MULTICLASS_MODEL = "python_results/multiclass/final_model_for_deployment.joblib"
METRICS_PATH     = "python_results/multiclass/performance_metrics.json"
OUTDIR           = "python_results/channel_importance_multiclass"
DATADIR          = "python_data"
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

ALL_CHANNELS = [
    'higacc_x_g', 'higacc_y_g', 'higacc_z_g',
    'inacc_x_ms', 'inacc_y_ms', 'inacc_z_ms',
    'rot_x_degs', 'rot_y_degs', 'rot_z_degs',
    'pressure_kpa'
]

CLASS_NAMES   = ['no_contact', 'indirect_strike', 'direct_strike']
N_PERMUTATIONS = 20


df = pd.read_csv(RAW_DATA, low_memory=False)
df = df[df["treatment"] != "400 (50%)"].copy()
df = df[df["roi"].str.contains("roi4_nadir", na=False)].copy()

# Multiclass labels
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
          "treatment":    "first"
      })
      .reset_index()
)

print(f"Files: {len(file_metadata)}")
for i, cn in enumerate(CLASS_NAMES):
    count = (file_metadata['strike_class'] == i).sum()
    print(f"  Class {i} ({cn:17s}): {count:3d} ({count/len(file_metadata)*100:.1f}%)")

# Extract time series
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

X_full = np.stack(X_full_list)
y      = file_metadata["strike_class"].values

print(f"Data shapes: X={X_full.shape}, y={y.shape}")

# Class weights for ablation training
class_weights_arr = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights_arr))
sample_weights_full = np.array([class_weight_dict[label] for label in y])

# Channel group colours
def get_channel_group(name):
    if 'higacc'    in name: return 'High Acceleration'
    elif 'inacc'   in name: return 'Internal Acceleration'
    elif 'rot'     in name: return 'Rotation'
    elif 'pressure' in name: return 'Pressure'
    return 'Other'

GROUP_COLORS = {
    'High Acceleration':     '#1f77b4',
    'Internal Acceleration': '#ff7f0e',
    'Rotation':              '#2ca02c',
    'Pressure':              '#d62728'
}


print(f"\nLoading trained multiclass model from: {MULTICLASS_MODEL}")

model = joblib.load(MULTICLASS_MODEL)

# Baseline: argmax of decision function scores
scores_base  = model.decision_function(X_full)
preds_base   = np.argmax(scores_base, axis=1)
baseline_accuracy = accuracy_score(y, preds_base)

# Per-class baseline recall
prec_base, rec_base, _, _ = precision_recall_fscore_support(
    y, preds_base, average=None, zero_division=0)

print(f"Baseline accuracy (training data): {baseline_accuracy:.4f}")
print(f"Per-class recall:")
for i, cn in enumerate(CLASS_NAMES):
    print(f"  {cn:17s}: {rec_base[i]:.4f}")
print(f"\nShuffling each channel {N_PERMUTATIONS} times...")

perm_results = []

for ch_idx in range(len(ALL_CHANNELS)):
    ch_name = ALL_CHANNELS[ch_idx]
    print(f"\n[{ch_idx+1}/{len(ALL_CHANNELS)}] {ch_name}")

    acc_drops          = []
    indirect_rec_drops = []

    for perm in tqdm(range(N_PERMUTATIONS), desc="  Permuting", leave=False):
        try:
            X_shuffled = X_full.copy()
            np.random.shuffle(X_shuffled[:, ch_idx, :])

            scores_shuf = model.decision_function(X_shuffled)
            preds_shuf  = np.argmax(scores_shuf, axis=1)

            acc_shuf = accuracy_score(y, preds_shuf)
            _, rec_shuf, _, _ = precision_recall_fscore_support(
                y, preds_shuf, average=None, zero_division=0, labels=[0, 1, 2])

            acc_drops.append(baseline_accuracy - acc_shuf)
            indirect_rec_drops.append(rec_base[1] - rec_shuf[1])

        except Exception as e:
            print(f"    Warning perm {perm}: {e}")
            continue

    if len(acc_drops) > 0:
        perm_results.append({
            'channel_idx':            ch_idx,
            'channel_name':           ch_name,
            'channel_group':          get_channel_group(ch_name),
            'mean_accuracy_drop':     np.mean(acc_drops),
            'std_accuracy_drop':      np.std(acc_drops),
            'mean_indirect_rec_drop': np.mean(indirect_rec_drops),
            'std_indirect_rec_drop':  np.std(indirect_rec_drops),
            'n_successful':           len(acc_drops)
        })
        print(f"  Overall drop:   {np.mean(acc_drops):.4f} ± {np.std(acc_drops):.4f}")
        print(f"  Indirect drop:  {np.mean(indirect_rec_drops):.4f} ± {np.std(indirect_rec_drops):.4f}")

perm_df = pd.DataFrame(perm_results).sort_values('mean_accuracy_drop', ascending=False)
perm_df['rank'] = range(1, len(perm_df) + 1)
total_drop = perm_df['mean_accuracy_drop'].sum()
perm_df['pct_contribution'] = (perm_df['mean_accuracy_drop'] / total_drop * 100) if total_drop > 0 else 0

print(f"{'Rank':<5} {'Channel':<20} {'Acc Drop':<12} {'Indirect Drop':<15}")
for _, row in perm_df.iterrows():
    print(f"{row['rank']:<5} {row['channel_name']:<20} "
          f"{row['mean_accuracy_drop']:>10.4f}   "
          f"{row['mean_indirect_rec_drop']:>12.4f}")


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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ablation_results = []
total_start = time.time()

for cfg_idx, cfg in enumerate(configurations):
    print(f"\n{'='*70}")
    print(f"Config {cfg_idx+1}/{len(configurations)}: {cfg['name']}")
    print(f"{'='*70}")

    ch_indices = [ALL_CHANNELS.index(ch) for ch in cfg['channels']]
    X_cfg      = X_full[:, ch_indices, :]

    oof_preds_cfg = np.zeros(len(y), dtype=int)
    fold_accs     = []

    cfg_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_cfg, y)):
        pipeline = make_pipeline(
            MiniRocket(random_state=42, n_jobs=-1),
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')
        )

        fold_weights = sample_weights_full[train_idx]
        pipeline.fit(X_cfg[train_idx], y[train_idx],
                     ridgeclassifiercv__sample_weight=fold_weights)

        scores = pipeline.decision_function(X_cfg[test_idx])
        preds  = np.argmax(scores, axis=1)

        oof_preds_cfg[test_idx] = preds
        fold_accs.append(accuracy_score(y[test_idx], preds))

    # Overall metrics
    cv_accuracy = accuracy_score(y, oof_preds_cfg)
    prec_all, rec_all, f1_all, support_all = precision_recall_fscore_support(
        y, oof_preds_cfg, average=None, zero_division=0, labels=[0, 1, 2])
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y, oof_preds_cfg, average='macro', zero_division=0)

    cm_cfg = confusion_matrix(y, oof_preds_cfg)
    cfg_time = time.time() - cfg_start

    ablation_results.append({
        'configuration':        cfg['name'],
        'description':          cfg['description'],
        'n_channels':           len(cfg['channels']),
        'cv_accuracy':          cv_accuracy,
        'cv_accuracy_std':      np.std(fold_accs),
        'macro_precision':      prec_macro,
        'macro_recall':         rec_macro,
        'macro_f1':             f1_macro,
        'no_contact_recall':    rec_all[0],
        'indirect_recall':      rec_all[1],
        'direct_recall':        rec_all[2],
        'confusion_matrix':     cm_cfg.tolist(),
        'training_time_sec':    cfg_time
    })

    print(f"\n  CV Accuracy:      {cv_accuracy:.4f} (±{np.std(fold_accs):.4f})")
    print(f"  Macro F1:         {f1_macro:.4f}")
    print(f"  No-contact recall:{rec_all[0]:.4f}")
    print(f"  Indirect recall:  {rec_all[1]:.4f}")
    print(f"  Direct recall:    {rec_all[2]:.4f}")
    print(f"  Time: {cfg_time:.1f}s")

print(f"\nTotal ablation time: {(time.time()-total_start)/60:.1f} minutes")

ablation_df = pd.DataFrame(ablation_results)
baseline_acc      = ablation_df[ablation_df['configuration']=='higacc_only']['cv_accuracy'].values[0]
baseline_indirect = ablation_df[ablation_df['configuration']=='higacc_only']['indirect_recall'].values[0]
ablation_df['accuracy_gain_vs_minimal']  = ablation_df['cv_accuracy']    - baseline_acc
ablation_df['indirect_gain_vs_minimal']  = ablation_df['indirect_recall'] - baseline_indirect


print(f"{'Config':<25} {'Ch':>3} {'Accuracy':>10} {'Indirect Recall':>16} {'Gain':>8}")
print("-" * 90)
for _, row in ablation_df.iterrows():
    gain = f"+{row['accuracy_gain_vs_minimal']*100:.1f}%" if row['accuracy_gain_vs_minimal'] > 0 else f"{row['accuracy_gain_vs_minimal']*100:.1f}%"
    print(f"{row['description']:<25} {row['n_channels']:>3} "
          f"{row['cv_accuracy']:>10.4f} {row['indirect_recall']:>16.4f} {gain:>8}")



plt.style.use('default')
config_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
config_labels = ['Minimal\n(3 ch)', 'Medium\n(4 ch)', 'Full\n(10 ch)']

# 1. Accuracy drop bar chart
fig, ax = plt.subplots(figsize=(10, 7))
bar_colors = [GROUP_COLORS[g] for g in perm_df['channel_group']]
ax.barh(perm_df['channel_name'], perm_df['mean_accuracy_drop'],
        xerr=perm_df['std_accuracy_drop'],
        color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
ax.set_xlabel('Mean Accuracy Drop', fontsize=12, fontweight='bold')
ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
ax.set_title(f'Channel Importance - Multiclass Model\n({N_PERMUTATIONS} permutations)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=g, alpha=0.8)
                   for g in GROUP_COLORS if g in perm_df['channel_group'].values]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/accuracy_drop_by_channel.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Indirect recall drop bar chart
perm_df_indirect = perm_df.sort_values('mean_indirect_rec_drop', ascending=False)
fig, ax = plt.subplots(figsize=(10, 7))
bar_colors_ind = [GROUP_COLORS[g] for g in perm_df_indirect['channel_group']]
ax.barh(perm_df_indirect['channel_name'], perm_df_indirect['mean_indirect_rec_drop'],
        xerr=perm_df_indirect['std_indirect_rec_drop'],
        color=bar_colors_ind, alpha=0.8, edgecolor='black', lw=0.5)
ax.set_xlabel('Mean Indirect Recall Drop', fontsize=12, fontweight='bold')
ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
ax.set_title(f'Channel Importance for Indirect Strike Detection\n({N_PERMUTATIONS} permutations)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/indirect_drop_by_channel.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Channel importance box plot
perm_raw = {ch: [] for ch in ALL_CHANNELS}
np.random.seed(42)
print("\nCollecting raw permutation data for box plot...")
for ch_idx, ch_name in enumerate(ALL_CHANNELS):
    for _ in range(N_PERMUTATIONS):
        try:
            X_shuffled = X_full.copy()
            np.random.shuffle(X_shuffled[:, ch_idx, :])
            scores_shuf = model.decision_function(X_shuffled)
            preds_shuf  = np.argmax(scores_shuf, axis=1)
            perm_raw[ch_name].append(baseline_accuracy - accuracy_score(y, preds_shuf))
        except:
            continue

ordered_channels = perm_df['channel_name'].tolist()
fig, ax = plt.subplots(figsize=(10, 7))
box_data   = [perm_raw[ch] for ch in ordered_channels]
box_colors = [GROUP_COLORS[get_channel_group(ch)] for ch in ordered_channels]
bp = ax.boxplot(box_data, labels=ordered_channels, vert=False,
                patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xlabel('Accuracy Drop', fontsize=12, fontweight='bold')
ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
ax.set_title(f'Distribution of Permutation Importance - Multiclass\n({N_PERMUTATIONS} permutations)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/channel_importance_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()


# 4. Permutation heatmap
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
ax.set_title('Permutation Importance Heatmap - Multiclass', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/permutation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()


# 5. Scatter: overall vs indirect importance
fig, ax = plt.subplots(figsize=(9, 7))
bar_colors_s = [GROUP_COLORS[g] for g in perm_df['channel_group']]
ax.scatter(perm_df['mean_accuracy_drop'], perm_df['mean_indirect_rec_drop'],
           s=100, c=bar_colors_s, edgecolors='black', zorder=5, alpha=0.8)
for _, row in perm_df.iterrows():
    ax.annotate(row['channel_name'],
                (row['mean_accuracy_drop'], row['mean_indirect_rec_drop']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.axvline(0, color='black', lw=0.8, linestyle='--')
ax.set_xlabel('Overall Accuracy Drop', fontsize=12, fontweight='bold')
ax.set_ylabel('Indirect Recall Drop', fontsize=12, fontweight='bold')
ax.set_title('Channel Importance: Overall vs Indirect (Multiclass)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/importance_scatter.png", dpi=300, bbox_inches='tight')
plt.close()


# 6. Configuration comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
metrics_to_plot = [
    ('cv_accuracy',     'Overall Accuracy'),
    ('macro_f1',        'Macro F1-Score'),
    ('indirect_recall', 'Indirect Recall')
]
for ax, (metric, title) in zip(axes, metrics_to_plot):
    values = ablation_df[metric].values
    bars   = ax.bar(config_labels, values, color=config_colors, alpha=0.8, edgecolor='black')
    ax.set_ylim([max(0, min(values)*0.9), 1.0])
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.suptitle('Sensor Configuration Comparison (Multiclass)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/configuration_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Ablation heatmap
fig, ax = plt.subplots(figsize=(10, 5))
heatmap_data = ablation_df[['cv_accuracy', 'macro_f1', 'no_contact_recall',
                             'indirect_recall', 'direct_recall']].set_index(ablation_df['description'])
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Score'})
ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study Heatmap (Multiclass)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/ablation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

perm_df.to_csv(f"{OUTDIR}/channel_importance_ranking.csv", index=False)
perm_df.to_csv(f"{OUTDIR}/channel_importance_summary.csv", index=False)

perm_detailed_rows = []
for ch_name, drops in perm_raw.items():
    for perm_n, drop_val in enumerate(drops):
        perm_detailed_rows.append({
            'channel_name':  ch_name,
            'channel_group': get_channel_group(ch_name),
            'permutation':   perm_n,
            'accuracy_drop': drop_val
        })
pd.DataFrame(perm_detailed_rows).to_csv(
    f"{OUTDIR}/permutation_importance_detailed.csv", index=False)

ablation_df.to_csv(f"{OUTDIR}/ablation_study_detailed.csv", index=False)

ablation_df[['configuration', 'description', 'n_channels', 'cv_accuracy',
             'macro_f1', 'no_contact_recall', 'indirect_recall',
             'direct_recall']].to_csv(f"{OUTDIR}/sensor_comparison_results.csv", index=False)

ablation_df.to_csv(f"{OUTDIR}/model_comparison.csv", index=False)

full_row    = ablation_df[ablation_df['configuration']=='full_suite'].iloc[0]
minimal_row = ablation_df[ablation_df['configuration']=='higacc_only'].iloc[0]

summary = {
    "model": "Multiclass (0=no_contact, 1=indirect, 2=direct)",
    "permutation_importance": {
        "n_permutations":      N_PERMUTATIONS,
        "baseline_accuracy":   float(baseline_accuracy),
        "most_important":      perm_df.iloc[0]['channel_name'],
        "least_important":     perm_df.iloc[-1]['channel_name'],
        "top_3_channels":      perm_df.head(3)['channel_name'].tolist()
    },
    "ablation_study": {
        "higacc_only_accuracy":          float(minimal_row['cv_accuracy']),
        "higacc_only_indirect_recall":   float(minimal_row['indirect_recall']),
        "full_suite_accuracy":           float(full_row['cv_accuracy']),
        "full_suite_indirect_recall":    float(full_row['indirect_recall']),
        "accuracy_gain_minimal_to_full": float(full_row['cv_accuracy'] - minimal_row['cv_accuracy']),
        "indirect_gain_minimal_to_full": float(full_row['indirect_recall'] - minimal_row['indirect_recall'])
    }
}

with open(f"{OUTDIR}/channel_importance_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Most important (overall):   {perm_df.iloc[0]['channel_name']} "
      f"(drop: {perm_df.iloc[0]['mean_accuracy_drop']:.4f})")
print(f"  Most important (indirect):  "
      f"{perm_df_indirect.iloc[0]['channel_name']} "
      f"(indirect drop: {perm_df_indirect.iloc[0]['mean_indirect_rec_drop']:.4f})")

print(f"  Minimal (3 ch):  {minimal_row['cv_accuracy']:.3f} accuracy, "
      f"{minimal_row['indirect_recall']:.3f} indirect recall")
print(f"  Full (10 ch):    {full_row['cv_accuracy']:.3f} accuracy, "
      f"{full_row['indirect_recall']:.3f} indirect recall")

print(f"\nOutputs: {OUTDIR}/")
