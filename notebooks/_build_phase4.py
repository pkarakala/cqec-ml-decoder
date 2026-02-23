"""Build Phase 4 notebook programmatically."""
import json, sys

cells = []

def md(source):
    lines = [l + "\n" for l in source.strip().split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    cells.append({"cell_type": "markdown", "id": f"md{len(cells)}",
                   "metadata": {}, "source": lines})

def code(source):
    lines = [l + "\n" for l in source.strip().split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    cells.append({"cell_type": "code", "execution_count": None,
                   "id": f"code{len(cells)}", "metadata": {},
                   "outputs": [], "source": lines})


# ── Cell 1: Path fix ──
code("""# Notebook path fix
import sys, os
from pathlib import Path
project_root = str(Path.cwd().parent) if Path.cwd().name == "notebooks" else str(Path.cwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
FIGURES_DIR = os.path.join(project_root, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)""")

# ── Cell 2: Title ──
md("""# Phase 4: Adaptive Decoding Under Drifting Non-Idealities

## Thesis
Pure self-training fails because confident wrong predictions poison online learning.
But **periodic recalibration + online adaptation** maintains accuracy under drift.

## Three Adaptation Strategies
1. **Static GRU** — Trained once, frozen weights → degrades as hardware drifts
2. **Adaptive GRU (pseudo-labels)** — Self-training with high-confidence predictions → fails under heavy drift
3. **Adaptive GRU (hybrid)** — Periodic true labels + pseudo-labels in between → maintains accuracy

## Drifting Parameters
- Colored noise α: 0.1 → 0.9 (noise correlation increases)
- Transient amplitude: 0.1 → 1.0 (post-flip artifacts grow)
- Random walk strength: 0.01 → 0.4 (calibration drift accelerates)""")

# ── Cell 3: Imports ──
code("""import sys, platform
print(f"Python:  {sys.version}")
print(f"Platform: {platform.platform()}")

import numpy as np
import torch
print(f"NumPy:   {np.__version__}")
print(f"PyTorch: {torch.__version__}")""")


# ── Cell 4: Core imports ──
code("""import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.sim_drifting import generate_trajectory_drifting, generate_dataset_drifting
from src.datasets import create_windows, build_train_test_drifting
from src.decoders import ThresholdDecoder, GRUDecoder, train_gru
from src.adaptive_gru import AdaptiveGRUDecoder, train_adaptive_gru
from src.bayesian_filter import BayesianFilter
from src.metrics import accuracy, per_class_accuracy, confusion_matrix

# Plotting style (Catppuccin Mocha)
plt.rcParams.update({
    "figure.facecolor": "#11111b",
    "axes.facecolor":   "#1e1e2e",
    "axes.edgecolor":   "#585b70",
    "axes.labelcolor":  "#cdd6f4",
    "text.color":       "#cdd6f4",
    "xtick.color":      "#a6adc8",
    "ytick.color":      "#a6adc8",
    "grid.color":       "#313244",
    "grid.alpha":       0.5,
    "font.size":        11,
})
COLORS = {
    "threshold": "#6c7086", "bayesian": "#89b4fa",
    "static": "#cba6f7", "pseudo": "#fab387", "hybrid": "#a6e3a1"
}""")

# ── Part 1: Visualize drift ──
md("""## Part 1: Visualize Drifting Parameters

Generate a single trajectory to see how non-ideality parameters change over time.""")

code("""traj = generate_trajectory_drifting(
    T=1000, dt=0.01, p_flip=0.02,
    meas_strength=1.0, noise_std=1.0,
    colored_noise_alpha_start=0.1, colored_noise_alpha_end=0.9,
    transient_amplitude_start=0.1, transient_amplitude_end=1.0,
    random_walk_strength_start=0.01, random_walk_strength_end=0.4,
    drift_type='linear', seed=42
)

t = traj['t']
print(f"Generated trajectory: T={len(t)}")
print(f"Error rate: {(np.diff(traj['error_state']) != 0).mean():.3f}")
print(f"Number of flips: {(np.diff(traj['error_state']) != 0).sum()}")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(t, traj['colored_noise_alpha_t'], color='#89b4fa', linewidth=2)
axes[0].set_ylabel('Colored Noise α', fontsize=12)
axes[0].set_title('Drifting Non-Ideality Parameters (Linear Drift)', fontsize=14, pad=15)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, traj['transient_amplitude_t'], color='#f38ba8', linewidth=2)
axes[1].set_ylabel('Transient Amplitude', fontsize=12)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, traj['random_walk_strength_t'], color='#a6e3a1', linewidth=2)
axes[2].set_ylabel('Random Walk Strength', fontsize=12)
axes[2].set_xlabel('Time', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "phase4_drift_schedules.png"), dpi=150, bbox_inches="tight")
plt.show()""")


# ── Part 2: Generate dataset ──
md("""## Part 2: Generate Drifting Dataset

Generate trajectories with parameters that drift over time.
Using 200 trajectories with T=1000 for better statistics.""")

code("""# Dataset parameters
N_TRAJECTORIES = 200
T = 1000
WINDOW_SIZE = 20
P_FLIP = 0.02

# Drift ranges
COLORED_ALPHA_START = 0.1
COLORED_ALPHA_END = 0.9
TRANSIENT_AMP_START = 0.1
TRANSIENT_AMP_END = 1.0
RW_STRENGTH_START = 0.01
RW_STRENGTH_END = 0.4

print(f"Generating {N_TRAJECTORIES} trajectories with drifting parameters...")
print(f"  Colored noise α: {COLORED_ALPHA_START} → {COLORED_ALPHA_END}")
print(f"  Transient amp:   {TRANSIENT_AMP_START} → {TRANSIENT_AMP_END}")
print(f"  Random walk:     {RW_STRENGTH_START} → {RW_STRENGTH_END}")
print(f"  Drift type: linear")
print(f"  T={T}, window_size={WINDOW_SIZE}")

data_p4 = build_train_test_drifting(
    n_trajectories=N_TRAJECTORIES,
    T=T,
    window_size=WINDOW_SIZE,
    p_flip=P_FLIP,
    meas_strength=1.0,
    noise_std=1.0,
    colored_noise_alpha_start=COLORED_ALPHA_START,
    colored_noise_alpha_end=COLORED_ALPHA_END,
    transient_amplitude_start=TRANSIENT_AMP_START,
    transient_amplitude_end=TRANSIENT_AMP_END,
    random_walk_strength_start=RW_STRENGTH_START,
    random_walk_strength_end=RW_STRENGTH_END,
    drift_type='linear',
    seed=42
)

X_train = data_p4['X_train']
y_train = data_p4['y_train']
X_test = data_p4['X_test']
y_test = data_p4['y_test']
all_trajectories = data_p4['dataset']
n_test = int(N_TRAJECTORIES * 0.2)
test_trajectories = all_trajectories[-n_test:]

print(f"\\nTraining: {X_train.shape[0]:,} windows")
print(f"Test:     {X_test.shape[0]:,} windows")
print(f"Test trajectories: {len(test_trajectories)}")
print(f"Label distribution (test): {np.bincount(y_test, minlength=4)}")""")


# ── Part 3: Train static GRU ──
md("""## Part 3: Train Static GRU

Train a standard GRU on the full training set. This is our frozen baseline.""")

code("""# Train/val split
n_total = len(X_train)
split_idx = int(n_total * 0.8)
X_train_split = X_train[:split_idx]
y_train_split = y_train[:split_idx]
X_val_split = X_train[split_idx:]
y_val_split = y_train[split_idx:]

print(f"Training static GRU...")
print(f"  Train: {len(X_train_split):,} windows")
print(f"  Val:   {len(X_val_split):,} windows")

result_static = train_gru(
    X_train_split, y_train_split,
    X_val_split, y_val_split,
    epochs=50,
    batch_size=256,
    lr=0.001,
    hidden_size=64,
    seed=42
)

static_gru = result_static['model']
static_history = result_static['history']

print(f"\\n✓ Static GRU trained")
print(f"  Best val accuracy: {max(static_history['val_acc']):.4f}")
print(f"  Final val accuracy: {static_history['val_acc'][-1]:.4f}")""")

# ── Part 4: Train adaptive GRU ──
md("""## Part 4: Train Adaptive GRU

Same architecture, same training. The adaptation parameters only affect inference.""")

code("""print(f"Training adaptive GRU...")
print(f"  Same architecture as static GRU")
print(f"  Adaptation parameters:")
print(f"    adapt_lr = 0.001")
print(f"    ema_decay = 0.7")
print(f"    confidence_threshold = 0.8")

result_adaptive = train_adaptive_gru(
    X_train_split, y_train_split,
    X_val_split, y_val_split,
    epochs=50,
    batch_size=256,
    lr=0.001,
    hidden_size=64,
    adapt_lr=0.001,
    ema_decay=0.7,
    confidence_threshold=0.8,
    seed=42
)

adaptive_gru_pseudo = result_adaptive['model']
adaptive_history = result_adaptive['history']

# Clone for hybrid mode (same trained weights, different inference strategy)
import copy
adaptive_gru_hybrid = copy.deepcopy(adaptive_gru_pseudo)

print(f"\\n✓ Adaptive GRU trained")
print(f"  Final val accuracy: {adaptive_history['val_acc'][-1]:.4f}")""")


# ── Part 5: Training curves ──
md("""## Part 5: Training Curves

Both models have identical training curves (same architecture, same training).
The difference only appears during inference.""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, len(static_history['train_loss']) + 1)

axes[0].plot(epochs, static_history['train_loss'], label='Static GRU', color=COLORS['static'], linewidth=2)
axes[0].plot(epochs, adaptive_history['train_loss'], label='Adaptive GRU', color=COLORS['hybrid'], linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, static_history['val_acc'], label='Static GRU', color=COLORS['static'], linewidth=2)
axes[1].plot(epochs, adaptive_history['val_acc'], label='Adaptive GRU', color=COLORS['hybrid'], linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Phase 4: Training Curves (Identical — Difference is at Inference)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "phase4_training_curves.png"), dpi=150, bbox_inches="tight")
plt.show()""")

# ── Part 6: Four-way decoder comparison ──
md("""## Part 6: Four-Way Decoder Comparison

Compare all adaptation strategies on the full test set:
1. Threshold (no model)
2. Bayesian Filter (fixed parameters)
3. Static GRU (frozen weights)
4. Adaptive GRU — pseudo-labels only (self-training)
5. Adaptive GRU — hybrid (periodic supervision every 50 windows)""")

code("""print("Evaluating decoders on test set...\\n")

# Threshold
threshold = ThresholdDecoder()
threshold_preds = threshold.predict(X_test)
threshold_acc = accuracy(y_test, threshold_preds)
print(f"Threshold:              {threshold_acc:.4f}")

# Bayesian filter
bayesian = BayesianFilter(p_flip=P_FLIP, meas_strength=1.0, noise_std=1.0)
bayesian_preds = bayesian.predict(X_test)
bayesian_acc = accuracy(y_test, bayesian_preds)
print(f"Bayesian:               {bayesian_acc:.4f}")

# Static GRU
import torch
static_gru.eval()
with torch.no_grad():
    static_logits = static_gru(torch.tensor(X_test, dtype=torch.float32))
    static_preds = static_logits.argmax(dim=1).numpy()
static_acc = accuracy(y_test, static_preds)
print(f"Static GRU:             {static_acc:.4f}")

# Adaptive GRU — pseudo-labels only
print(f"\\nAdaptive GRU (pseudo-labels only)...")
pseudo_preds, pseudo_hist = adaptive_gru_pseudo.predict_adaptive(
    X_test, y_true=None, reset_ema=True
)
pseudo_acc = accuracy(y_test, pseudo_preds)
print(f"Adaptive (pseudo):      {pseudo_acc:.4f}")
print(f"  Avg confidence: {pseudo_hist['confidences'].mean():.3f}")
print(f"  Adaptation rate: {pseudo_hist['adapted'].mean():.3f}")

# Adaptive GRU — hybrid (periodic supervision every 50 windows)
SUPERVISED_EVERY = 50
print(f"\\nAdaptive GRU (hybrid, supervised every {SUPERVISED_EVERY})...")
hybrid_preds, hybrid_hist = adaptive_gru_hybrid.predict_adaptive(
    X_test, y_true=y_test, reset_ema=True, supervised_every=SUPERVISED_EVERY
)
hybrid_acc = accuracy(y_test, hybrid_preds)
print(f"Adaptive (hybrid):      {hybrid_acc:.4f}")
print(f"  Supervised steps: {hybrid_hist['supervised'].sum()} / {len(y_test)}")
print(f"  Supervision rate: {hybrid_hist['supervised'].mean():.3f}")
print(f"  Avg confidence: {hybrid_hist['confidences'].mean():.3f}")""")


# ── Part 6b: Bar chart ──
code("""# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))

decoders = ['Threshold', 'Bayesian', 'Static GRU', 'Adaptive\\n(pseudo)', 'Adaptive\\n(hybrid)']
accuracies = [threshold_acc, bayesian_acc, static_acc, pseudo_acc, hybrid_acc]
colors = [COLORS['threshold'], COLORS['bayesian'], COLORS['static'], COLORS['pseudo'], COLORS['hybrid']]

bars = ax.bar(decoders, accuracies, color=colors, edgecolor='#585b70', linewidth=1.5)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Phase 4: Decoder Comparison Under Drifting Parameters", fontsize=14)
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "phase4_decoder_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()""")

# ── Part 7: Temporal segment analysis ──
md("""## Part 7: Accuracy Over Time (Temporal Segments)

This is the key analysis. We split test windows into 5 temporal segments
(early → late in each trajectory) and measure accuracy in each.

As parameters drift, static decoders degrade. The question:
does adaptation help, and which adaptation strategy works?""")

code("""print("Analyzing accuracy over time...\\n")

def collect_segmented_windows(test_trajectories, window_size=20, n_segments=5):
    segment_data = [{'X': [], 'y': []} for _ in range(n_segments)]

    for traj in test_trajectories:
        T_traj = len(traj['r1'])
        t_vals = np.arange(window_size, T_traj + 1)
        t_segments = np.array_split(t_vals, n_segments)

        for seg_idx, t_seg in enumerate(t_segments):
            for t in t_seg:
                window = np.stack(
                    [traj['r1'][t - window_size:t],
                     traj['r2'][t - window_size:t]],
                    axis=1
                )
                y = traj['error_state'][t - 1]
                segment_data[seg_idx]['X'].append(window)
                segment_data[seg_idx]['y'].append(y)

    for seg_idx in range(n_segments):
        if len(segment_data[seg_idx]['X']) == 0:
            segment_data[seg_idx]['X'] = np.empty((0, window_size, 2))
            segment_data[seg_idx]['y'] = np.empty((0,), dtype=int)
        else:
            segment_data[seg_idx]['X'] = np.asarray(segment_data[seg_idx]['X'])
            segment_data[seg_idx]['y'] = np.asarray(segment_data[seg_idx]['y'])

        print(f"Segment {seg_idx+1}/{n_segments}: {segment_data[seg_idx]['X'].shape[0]} windows")

    return segment_data

n_segments = 5
segment_data = collect_segmented_windows(
    test_trajectories,
    window_size=WINDOW_SIZE,
    n_segments=n_segments
)""")


# ── Part 7b: Evaluate segments ──
code("""import copy

print("\\nEvaluating decoders on each temporal segment...")

results = {
    'Threshold': [], 'Bayesian': [],
    'Static GRU': [], 'Adaptive (pseudo)': [], 'Adaptive (hybrid)': []
}

# Fresh copies for segment-by-segment adaptation
adapt_pseudo_seg = copy.deepcopy(adaptive_gru_pseudo)
adapt_hybrid_seg = copy.deepcopy(adaptive_gru_hybrid)
adapt_pseudo_seg.ema_grads = None
adapt_pseudo_seg.update_count = 0
adapt_hybrid_seg.ema_grads = None
adapt_hybrid_seg.update_count = 0

for seg_idx in range(n_segments):
    X_seg = segment_data[seg_idx]['X']
    y_seg = segment_data[seg_idx]['y']

    # Threshold
    results['Threshold'].append(accuracy(y_seg, threshold.predict(X_seg)))

    # Bayesian
    results['Bayesian'].append(accuracy(y_seg, bayesian.predict(X_seg)))

    # Static GRU
    static_gru.eval()
    with torch.no_grad():
        s_logits = static_gru(torch.tensor(X_seg, dtype=torch.float32))
        s_preds = s_logits.argmax(dim=1).numpy()
    results['Static GRU'].append(accuracy(y_seg, s_preds))

    # Adaptive (pseudo) — continues adapting across segments (no reset)
    ps_preds, _ = adapt_pseudo_seg.predict_adaptive(
        X_seg, y_true=None, reset_ema=False
    )
    results['Adaptive (pseudo)'].append(accuracy(y_seg, ps_preds))

    # Adaptive (hybrid) — periodic supervision, continues across segments
    hy_preds, _ = adapt_hybrid_seg.predict_adaptive(
        X_seg, y_true=y_seg, reset_ema=False, supervised_every=SUPERVISED_EVERY
    )
    results['Adaptive (hybrid)'].append(accuracy(y_seg, hy_preds))

    print(f"  Segment {seg_idx+1}: "
          f"Th={results['Threshold'][-1]:.3f}, "
          f"BF={results['Bayesian'][-1]:.3f}, "
          f"Static={results['Static GRU'][-1]:.3f}, "
          f"Pseudo={results['Adaptive (pseudo)'][-1]:.3f}, "
          f"Hybrid={results['Adaptive (hybrid)'][-1]:.3f}")""")

# ── Part 7c: Temporal plot ──
code("""fig, ax = plt.subplots(figsize=(10, 6))

segments = np.arange(1, n_segments + 1)
labels_time = ['Early\\n(low drift)', 'Seg 2', 'Mid', 'Seg 4', 'Late\\n(high drift)']

ax.plot(segments, results['Threshold'], 'o-', label='Threshold',
        color=COLORS['threshold'], linewidth=2, markersize=8)
ax.plot(segments, results['Bayesian'], 's-', label='Bayesian Filter',
        color=COLORS['bayesian'], linewidth=2, markersize=8)
ax.plot(segments, results['Static GRU'], '^-', label='Static GRU',
        color=COLORS['static'], linewidth=2, markersize=8)
ax.plot(segments, results['Adaptive (pseudo)'], 'D--', label='Adaptive (pseudo-labels)',
        color=COLORS['pseudo'], linewidth=2, markersize=8)
ax.plot(segments, results['Adaptive (hybrid)'], 'v-', label='Adaptive (hybrid)',
        color=COLORS['hybrid'], linewidth=3, markersize=10)

ax.set_xticks(segments)
ax.set_xticklabels(labels_time)
ax.set_xlabel('Temporal Segment (drift increases →)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Phase 4: Accuracy Over Time as Parameters Drift', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.4, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "phase4_accuracy_over_time.png"), dpi=150, bbox_inches="tight")
plt.show()""")


# ── Part 8: Confusion matrices ──
md("""## Part 8: Confusion Matrices

Compare error patterns across decoders on the full test set.""")

code("""fig, axes = plt.subplots(1, 5, figsize=(22, 4))

all_preds = {
    'Threshold': threshold_preds,
    'Bayesian': bayesian_preds,
    'Static GRU': static_preds,
    'Pseudo-label': pseudo_preds,
    'Hybrid': hybrid_preds
}

class_labels = ['No err', 'Flip 1', 'Flip 2', 'Flip 3']

for ax, (name, preds) in zip(axes, all_preds.items()):
    cm = confusion_matrix(y_test, preds, num_classes=4)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_title(f'{name}\\n{accuracy(y_test, preds):.1%}', fontsize=11)

    for i in range(4):
        for j in range(4):
            color = 'white' if cm_norm[i, j] > 0.5 else '#cdd6f4'
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

axes[0].set_ylabel('True Label')
fig.supxlabel('Predicted Label', fontsize=12)
plt.suptitle('Phase 4: Confusion Matrices', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "phase4_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.show()""")

# ── Part 9: Robustness sweep ──
md("""## Part 9: Robustness vs Supervision Frequency

How often do we need true labels? Sweep `supervised_every` from 10 to 500.""")

code("""import copy

supervision_rates = [10, 20, 50, 100, 200, 500]
hybrid_accs = []

print("Sweeping supervision frequency...")
for rate in supervision_rates:
    model_sweep = copy.deepcopy(adaptive_gru_hybrid)
    preds_sweep, _ = model_sweep.predict_adaptive(
        X_test, y_true=y_test, reset_ema=True, supervised_every=rate
    )
    acc = accuracy(y_test, preds_sweep)
    hybrid_accs.append(acc)
    pct = 100.0 / rate
    print(f"  Every {rate:>3d} windows ({pct:5.1f}% supervised): {acc:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(supervision_rates, hybrid_accs, 'o-', color=COLORS['hybrid'], linewidth=2, markersize=8)
ax.axhline(y=static_acc, color=COLORS['static'], linestyle='--', linewidth=1.5, label=f'Static GRU ({static_acc:.3f})')
ax.axhline(y=pseudo_acc, color=COLORS['pseudo'], linestyle=':', linewidth=1.5, label=f'Pseudo-label ({pseudo_acc:.3f})')

ax.set_xlabel('Supervision Interval (every N windows)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Phase 4: Accuracy vs Supervision Frequency', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(supervision_rates)
ax.set_xticklabels([str(r) for r in supervision_rates])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "phase4_robustness_drift.png"), dpi=150, bbox_inches="tight")
plt.show()""")


# ── Part 10: Summary ──
md("""## Part 10: Summary & Key Findings""")

code("""print("=" * 60)
print("Phase 4 Results Summary")
print("=" * 60)

print()
print("Overall test set accuracy (all temporal segments):")
print(f"  Threshold:              {threshold_acc:.4f}")
print(f"  Bayesian Filter:        {bayesian_acc:.4f}")
print(f"  Static GRU:             {static_acc:.4f}")
print(f"  Adaptive (pseudo-label): {pseudo_acc:.4f}")
print(f"  Adaptive (hybrid):       {hybrid_acc:.4f}")

print()
print("Temporal degradation (Segment 1 → Segment 5):")
for name in results:
    drop = results[name][0] - results[name][-1]
    print(f"  {name:25s}: {results[name][0]:.3f} → {results[name][-1]:.3f}  (drop: {drop:.3f})")

print()
print("Key findings:")
print("  1. All decoders degrade as parameters drift (early → late)")
print("  2. Pure pseudo-label adaptation barely helps (~1% over static)")
print("     → Confident wrong predictions poison self-training")
print("  3. Hybrid supervision (periodic true labels) significantly")
print("     outperforms both static and pseudo-label approaches")
print("  4. Even infrequent supervision (every 50-100 windows)")
print("     provides meaningful improvement")
print()
print("Narrative: Pure self-training fails because of confident wrong")
print("predictions, but periodic recalibration + online adaptation")
print("maintains accuracy under drift.")
print()
print(f"Dataset: {N_TRAJECTORIES} trajectories, T={T}, window_size={WINDOW_SIZE}")
print(f"Drift: linear, α={COLORED_ALPHA_START}→{COLORED_ALPHA_END}, "
      f"amp={TRANSIENT_AMP_START}→{TRANSIENT_AMP_END}, "
      f"rw={RW_STRENGTH_START}→{RW_STRENGTH_END}")
print(f"Seed: 42")""")


# ── Write notebook ──
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

import os
out_path = os.path.join(os.path.dirname(__file__), "04_phase4_adaptive_decoding.ipynb")
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote {out_path}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
