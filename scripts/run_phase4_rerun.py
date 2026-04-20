"""Phase 4 re-run with tuned params. Saves figures to outputs/figures/ and
results to outputs/phase4_results.json so slides/README can be filled in.

Matches the notebook pipeline from notebooks/_build_phase4.py but runs as a
standalone script with the tuned (adapt_lr=0.005, ema_decay=0.5,
supervised_every=20) parameters and N_TRAJECTORIES, T per the re-run spec.
"""
import os
import sys
import json
import copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.sim_drifting import generate_trajectory_drifting
from src.datasets import build_train_test_drifting
from src.decoders import ThresholdDecoder, train_gru
from src.adaptive_gru import train_adaptive_gru
from src.bayesian_filter import BayesianFilter
from src.metrics import accuracy, confusion_matrix

# ── Runtime knobs ────────────────────────────────────────────
# Notebook spec is N=200, T=1000. To keep CPU-only wall time manageable we
# allow an override via env vars but default to the spec.
N_TRAJECTORIES = int(os.environ.get("P4_N", 200))
T = int(os.environ.get("P4_T", 1000))
EPOCHS = int(os.environ.get("P4_EPOCHS", 50))
WINDOW_SIZE = 20
P_FLIP = 0.02

# Drift ranges
COLORED_ALPHA_START, COLORED_ALPHA_END = 0.1, 0.9
TRANSIENT_AMP_START, TRANSIENT_AMP_END = 0.1, 1.0
RW_STRENGTH_START, RW_STRENGTH_END = 0.01, 0.4

# Tuned adaptation params (the re-run knobs)
ADAPT_LR = 0.005
EMA_DECAY = 0.5
SUPERVISED_EVERY = 20

COLORS = {
    "threshold": "#6c7086", "bayesian": "#89b4fa",
    "static": "#cba6f7", "pseudo": "#fab387", "hybrid": "#a6e3a1",
}
plt.rcParams.update({
    "figure.facecolor": "#11111b", "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#585b70", "axes.labelcolor": "#cdd6f4",
    "text.color": "#cdd6f4", "xtick.color": "#a6adc8",
    "ytick.color": "#a6adc8", "grid.color": "#313244", "grid.alpha": 0.5,
    "font.size": 11,
})

print(f"=== Phase 4 re-run (N={N_TRAJECTORIES}, T={T}, epochs={EPOCHS}) ===")

# ─── Part 1: Drift visualization ─────────────────────────────
traj_vis = generate_trajectory_drifting(
    T=T, dt=0.01, p_flip=P_FLIP, meas_strength=1.0, noise_std=1.0,
    colored_noise_alpha_start=COLORED_ALPHA_START,
    colored_noise_alpha_end=COLORED_ALPHA_END,
    transient_amplitude_start=TRANSIENT_AMP_START,
    transient_amplitude_end=TRANSIENT_AMP_END,
    random_walk_strength_start=RW_STRENGTH_START,
    random_walk_strength_end=RW_STRENGTH_END,
    drift_type="linear", seed=42,
)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axes[0].plot(traj_vis["t"], traj_vis["colored_noise_alpha_t"], color="#89b4fa", linewidth=2)
axes[0].set_ylabel("Colored Noise α")
axes[0].set_title("Drifting Non-Ideality Parameters (Linear Drift)", pad=15)
axes[0].grid(True, alpha=0.3)
axes[1].plot(traj_vis["t"], traj_vis["transient_amplitude_t"], color="#f38ba8", linewidth=2)
axes[1].set_ylabel("Transient Amplitude")
axes[1].grid(True, alpha=0.3)
axes[2].plot(traj_vis["t"], traj_vis["random_walk_strength_t"], color="#a6e3a1", linewidth=2)
axes[2].set_ylabel("Random Walk Strength")
axes[2].set_xlabel("Time")
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase4_drift_schedules.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Part 2: Dataset ─────────────────────────────────────────
print("Building drifting dataset...")
data = build_train_test_drifting(
    n_trajectories=N_TRAJECTORIES, T=T, window_size=WINDOW_SIZE, p_flip=P_FLIP,
    meas_strength=1.0, noise_std=1.0,
    colored_noise_alpha_start=COLORED_ALPHA_START,
    colored_noise_alpha_end=COLORED_ALPHA_END,
    transient_amplitude_start=TRANSIENT_AMP_START,
    transient_amplitude_end=TRANSIENT_AMP_END,
    random_walk_strength_start=RW_STRENGTH_START,
    random_walk_strength_end=RW_STRENGTH_END,
    drift_type="linear", seed=42,
)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
all_trajectories = data["dataset"]
n_test = int(N_TRAJECTORIES * 0.2)
test_trajectories = all_trajectories[-n_test:]
print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}  TestTraj: {len(test_trajectories)}")

# Train/val split within training set
n_total = len(X_train)
split = int(n_total * 0.8)
X_tr, y_tr = X_train[:split], y_train[:split]
X_va, y_va = X_train[split:], y_train[split:]

# ─── Part 3: Static GRU ──────────────────────────────────────
print("Training static GRU...")
res_static = train_gru(
    X_tr, y_tr, X_va, y_va,
    epochs=EPOCHS, batch_size=256, lr=0.001, hidden_size=64, seed=42,
)
static_gru = res_static["model"]
static_history = res_static["history"]
print(f"  static final val_acc = {static_history['val_acc'][-1]:.4f}")

# ─── Part 4: Adaptive GRU (identical training, tuned adapt params) ──
print("Training adaptive GRU (pseudo + hybrid share initial weights)...")
res_adapt = train_adaptive_gru(
    X_tr, y_tr, X_va, y_va,
    epochs=EPOCHS, batch_size=256, lr=0.001, hidden_size=64,
    adapt_lr=ADAPT_LR, ema_decay=EMA_DECAY,
    confidence_threshold=0.8, seed=42,
)
adaptive_gru_pseudo = res_adapt["model"]
adaptive_history = res_adapt["history"]
adaptive_gru_hybrid = copy.deepcopy(adaptive_gru_pseudo)

# ─── Training curves ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(static_history["train_loss"]) + 1)
axes[0].plot(epochs_range, static_history["train_loss"], label="Static GRU",
             color=COLORS["static"], linewidth=2)
axes[0].plot(epochs_range, adaptive_history["train_loss"], label="Adaptive GRU",
             color=COLORS["hybrid"], linewidth=2, linestyle="--")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(epochs_range, static_history["val_acc"], label="Static GRU",
             color=COLORS["static"], linewidth=2)
axes[1].plot(epochs_range, adaptive_history["val_acc"], label="Adaptive GRU",
             color=COLORS["hybrid"], linewidth=2, linestyle="--")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Accuracy")
axes[1].set_title("Validation Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle("Phase 4: Training Curves (Identical — Difference is at Inference)", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase4_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Part 5: Five-decoder comparison on full test set ────────
print("Evaluating decoders...")
threshold = ThresholdDecoder()
threshold_preds = threshold.predict(X_test)
threshold_acc = accuracy(y_test, threshold_preds)
print(f"  Threshold:   {threshold_acc:.4f}")

bayesian = BayesianFilter(p_flip=P_FLIP, meas_strength=1.0, noise_std=1.0)
bayesian_preds = bayesian.predict(X_test)
bayesian_acc = accuracy(y_test, bayesian_preds)
print(f"  Bayesian:    {bayesian_acc:.4f}")

static_gru.eval()
with torch.no_grad():
    static_preds = static_gru(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
static_acc = accuracy(y_test, static_preds)
print(f"  Static GRU:  {static_acc:.4f}")

print("  Running adaptive (pseudo-label)...")
pseudo_preds, pseudo_hist = adaptive_gru_pseudo.predict_adaptive(
    X_test, y_true=None, reset_ema=True
)
pseudo_acc = accuracy(y_test, pseudo_preds)
pseudo_conf = float(pseudo_hist["confidences"].mean())
print(f"  Pseudo:      {pseudo_acc:.4f}  (avg conf {pseudo_conf:.3f})")

print(f"  Running adaptive (hybrid, every {SUPERVISED_EVERY})...")
hybrid_preds, hybrid_hist = adaptive_gru_hybrid.predict_adaptive(
    X_test, y_true=y_test, reset_ema=True, supervised_every=SUPERVISED_EVERY
)
hybrid_acc = accuracy(y_test, hybrid_preds)
print(f"  Hybrid:      {hybrid_acc:.4f}")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
names = ["Threshold", "Bayesian", "Static GRU", "Adaptive\n(pseudo)", "Adaptive\n(hybrid)"]
accs = [threshold_acc, bayesian_acc, static_acc, pseudo_acc, hybrid_acc]
cols = [COLORS["threshold"], COLORS["bayesian"], COLORS["static"], COLORS["pseudo"], COLORS["hybrid"]]
bars = ax.bar(names, accs, color=cols, edgecolor="#585b70", linewidth=1.5)
for b, v in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{v:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
ax.set_title(f"Phase 4: Decoder Comparison Under Drifting Parameters (N={N_TRAJECTORIES})")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase4_decoder_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Part 6: Temporal segments ───────────────────────────────
print("Temporal segment analysis...")

def collect_segmented_windows(trajs, window_size=20, n_segments=5):
    data = [{"X": [], "y": []} for _ in range(n_segments)]
    for tr in trajs:
        T_tr = len(tr["r1"])
        t_vals = np.arange(window_size, T_tr + 1)
        for seg_idx, t_seg in enumerate(np.array_split(t_vals, n_segments)):
            for t in t_seg:
                w = np.stack([tr["r1"][t - window_size:t],
                              tr["r2"][t - window_size:t]], axis=1)
                data[seg_idx]["X"].append(w)
                data[seg_idx]["y"].append(tr["error_state"][t - 1])
    for d in data:
        d["X"] = np.asarray(d["X"]) if d["X"] else np.empty((0, window_size, 2))
        d["y"] = np.asarray(d["y"], dtype=int) if len(d["y"]) else np.empty((0,), dtype=int)
    return data


n_segments = 5
segment_data = collect_segmented_windows(test_trajectories, WINDOW_SIZE, n_segments)

results = {"Threshold": [], "Bayesian": [], "Static GRU": [],
           "Adaptive (pseudo)": [], "Adaptive (hybrid)": []}
seg_pseudo_confs = []

# Fresh copies that continuously adapt across segments
adapt_pseudo_seg = copy.deepcopy(adaptive_gru_pseudo)
adapt_hybrid_seg = copy.deepcopy(adaptive_gru_hybrid)
adapt_pseudo_seg.ema_grads = None; adapt_pseudo_seg.update_count = 0
adapt_hybrid_seg.ema_grads = None; adapt_hybrid_seg.update_count = 0

for seg_idx in range(n_segments):
    Xs, ys = segment_data[seg_idx]["X"], segment_data[seg_idx]["y"]
    results["Threshold"].append(accuracy(ys, threshold.predict(Xs)))
    results["Bayesian"].append(accuracy(ys, bayesian.predict(Xs)))
    static_gru.eval()
    with torch.no_grad():
        sp = static_gru(torch.tensor(Xs, dtype=torch.float32)).argmax(dim=1).numpy()
    results["Static GRU"].append(accuracy(ys, sp))
    ps, ph = adapt_pseudo_seg.predict_adaptive(Xs, y_true=None, reset_ema=False)
    results["Adaptive (pseudo)"].append(accuracy(ys, ps))
    seg_pseudo_confs.append(float(ph["confidences"].mean()))
    hy, _ = adapt_hybrid_seg.predict_adaptive(Xs, y_true=ys, reset_ema=False,
                                              supervised_every=SUPERVISED_EVERY)
    results["Adaptive (hybrid)"].append(accuracy(ys, hy))
    print(f"  Seg {seg_idx+1}: Th={results['Threshold'][-1]:.3f}  "
          f"BF={results['Bayesian'][-1]:.3f}  St={results['Static GRU'][-1]:.3f}  "
          f"Ps={results['Adaptive (pseudo)'][-1]:.3f}  Hy={results['Adaptive (hybrid)'][-1]:.3f}")

# Temporal plot
fig, ax = plt.subplots(figsize=(10, 6))
segments = np.arange(1, n_segments + 1)
labels_time = ["Early\n(low drift)", "Seg 2", "Mid", "Seg 4", "Late\n(high drift)"]
ax.plot(segments, results["Threshold"], "o-", label="Threshold", color=COLORS["threshold"],
        linewidth=2, markersize=8)
ax.plot(segments, results["Bayesian"], "s-", label="Bayesian Filter", color=COLORS["bayesian"],
        linewidth=2, markersize=8)
ax.plot(segments, results["Static GRU"], "^-", label="Static GRU", color=COLORS["static"],
        linewidth=2, markersize=8)
ax.plot(segments, results["Adaptive (pseudo)"], "D--", label="Adaptive (pseudo-labels)",
        color=COLORS["pseudo"], linewidth=2, markersize=8)
ax.plot(segments, results["Adaptive (hybrid)"], "v-", label="Adaptive (hybrid)",
        color=COLORS["hybrid"], linewidth=3, markersize=10)
ax.set_xticks(segments); ax.set_xticklabels(labels_time)
ax.set_xlabel("Temporal Segment (drift increases →)")
ax.set_ylabel("Accuracy")
ax.set_title("Phase 4: Accuracy Over Time as Parameters Drift")
ax.legend(loc="lower left"); ax.grid(True, alpha=0.3); ax.set_ylim(0.3, 1.0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase4_accuracy_over_time.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Part 7: Confusion matrices ──────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
all_preds = {"Threshold": threshold_preds, "Bayesian": bayesian_preds,
             "Static GRU": static_preds, "Pseudo-label": pseudo_preds, "Hybrid": hybrid_preds}
class_labels = ["No err", "Flip 1", "Flip 2", "Flip 3"]
for ax, (name, preds) in zip(axes, all_preds.items()):
    cm = confusion_matrix(y_test, preds)
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_title(f"{name}\n{accuracy(y_test, preds):.1%}", fontsize=11)
    for i in range(4):
        for j in range(4):
            color = "white" if cm_n[i, j] > 0.5 else "#cdd6f4"
            ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
axes[0].set_ylabel("True Label")
fig.supxlabel("Predicted Label", fontsize=12)
plt.suptitle("Phase 4: Confusion Matrices", fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase4_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Part 8: Supervision sweep ───────────────────────────────
print("Supervision sweep...")
supervision_rates = [10, 20, 50, 100, 200, 500]
hybrid_accs = []
for rate in supervision_rates:
    m = copy.deepcopy(adaptive_gru_hybrid)
    ps, _ = m.predict_adaptive(X_test, y_true=y_test, reset_ema=True, supervised_every=rate)
    a = accuracy(y_test, ps)
    hybrid_accs.append(a)
    print(f"  every {rate}: {a:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(supervision_rates, hybrid_accs, "o-", color=COLORS["hybrid"], linewidth=2, markersize=8)
ax.axhline(y=static_acc, color=COLORS["static"], linestyle="--", linewidth=1.5,
           label=f"Static GRU ({static_acc:.3f})")
ax.axhline(y=pseudo_acc, color=COLORS["pseudo"], linestyle=":", linewidth=1.5,
           label=f"Pseudo-label ({pseudo_acc:.3f})")
ax.set_xlabel("Supervision Interval (every N windows)")
ax.set_ylabel("Accuracy")
ax.set_title("Phase 4: Accuracy vs Supervision Frequency")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xscale("log")
ax.set_xticks(supervision_rates)
ax.set_xticklabels([str(r) for r in supervision_rates])
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase4_robustness_drift.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Serialize results ───────────────────────────────────────
out = {
    "config": {
        "N_TRAJECTORIES": N_TRAJECTORIES, "T": T, "EPOCHS": EPOCHS,
        "WINDOW_SIZE": WINDOW_SIZE, "adapt_lr": ADAPT_LR,
        "ema_decay": EMA_DECAY, "supervised_every": SUPERVISED_EVERY,
    },
    "overall": {
        "threshold": threshold_acc, "bayesian": bayesian_acc,
        "static": static_acc, "pseudo": pseudo_acc, "hybrid": hybrid_acc,
    },
    "segments": {k: [float(x) for x in v] for k, v in results.items()},
    "pseudo_avg_confidence": pseudo_conf,
    "pseudo_seg_confidences": [float(c) for c in seg_pseudo_confs],
    "pseudo_seg_accuracies": results["Adaptive (pseudo)"],
    "supervision_sweep": {
        "rates": supervision_rates,
        "accuracies": [float(a) for a in hybrid_accs],
    },
    "static_history_peak_val_acc": max(static_history["val_acc"]),
}
out_path = ROOT / "outputs" / "phase4_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {out_path}")
print("=== Phase 4 re-run complete ===")
