"""Smoke test for Phase 4 notebook logic with tiny dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import copy

from src.sim_drifting import generate_trajectory_drifting
from src.datasets import build_train_test_drifting, create_windows
from src.decoders import ThresholdDecoder, GRUDecoder, train_gru
from src.adaptive_gru import AdaptiveGRUDecoder, train_adaptive_gru
from src.bayesian_filter import BayesianFilter
from src.metrics import accuracy, confusion_matrix

print("=" * 50)
print("Phase 4 Smoke Test (tiny dataset)")
print("=" * 50)

# ── Tiny dataset ──
N_TRAJ = 20
T = 200
WINDOW_SIZE = 20
P_FLIP = 0.02
SUPERVISED_EVERY = 10

print(f"\n1. Generating {N_TRAJ} trajectories (T={T})...")
data = build_train_test_drifting(
    n_trajectories=N_TRAJ, T=T, window_size=WINDOW_SIZE, p_flip=P_FLIP,
    meas_strength=1.0, noise_std=1.0,
    colored_noise_alpha_start=0.1, colored_noise_alpha_end=0.9,
    transient_amplitude_start=0.1, transient_amplitude_end=1.0,
    random_walk_strength_start=0.01, random_walk_strength_end=0.4,
    drift_type='linear', seed=42
)

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
all_trajectories = data['dataset']
n_test = int(N_TRAJ * 0.2)
test_trajectories = all_trajectories[-n_test:]
print(f"   Train: {X_train.shape[0]} windows, Test: {X_test.shape[0]} windows")
print(f"   Test trajectories: {len(test_trajectories)}")

# ── Train static GRU ──
print("\n2. Training static GRU (5 epochs)...")
split = int(len(X_train) * 0.8)
result_static = train_gru(
    X_train[:split], y_train[:split], X_train[split:], y_train[split:],
    epochs=5, batch_size=64, lr=0.001, hidden_size=32, seed=42
)
static_gru = result_static['model']
print(f"   Final val acc: {result_static['history']['val_acc'][-1]:.4f}")

# ── Train adaptive GRU ──
print("\n3. Training adaptive GRU (5 epochs)...")
result_adapt = train_adaptive_gru(
    X_train[:split], y_train[:split], X_train[split:], y_train[split:],
    epochs=5, batch_size=64, lr=0.001, hidden_size=32,
    adapt_lr=0.001, ema_decay=0.7, confidence_threshold=0.8, seed=42
)
adapt_pseudo = result_adapt['model']
adapt_hybrid = copy.deepcopy(adapt_pseudo)
print(f"   Final val acc: {result_adapt['history']['val_acc'][-1]:.4f}")

# ── Evaluate all decoders ──
print("\n4. Evaluating decoders...")

threshold = ThresholdDecoder()
th_acc = accuracy(y_test, threshold.predict(X_test))

bayesian = BayesianFilter(p_flip=P_FLIP, meas_strength=1.0, noise_std=1.0)
bf_acc = accuracy(y_test, bayesian.predict(X_test))

static_gru.eval()
with torch.no_grad():
    s_preds = static_gru(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
static_acc = accuracy(y_test, s_preds)

pseudo_preds, pseudo_hist = adapt_pseudo.predict_adaptive(X_test, y_true=None, reset_ema=True)
pseudo_acc = accuracy(y_test, pseudo_preds)

hybrid_preds, hybrid_hist = adapt_hybrid.predict_adaptive(
    X_test, y_true=y_test, reset_ema=True, supervised_every=SUPERVISED_EVERY
)
hybrid_acc = accuracy(y_test, hybrid_preds)

print(f"   Threshold:         {th_acc:.4f}")
print(f"   Bayesian:          {bf_acc:.4f}")
print(f"   Static GRU:        {static_acc:.4f}")
print(f"   Adaptive (pseudo): {pseudo_acc:.4f}")
print(f"   Adaptive (hybrid): {hybrid_acc:.4f}")
print(f"   Supervised steps:  {hybrid_hist['supervised'].sum()}/{len(y_test)}")


# ── Temporal segments ──
print("\n5. Temporal segment analysis...")

def collect_segmented_windows(test_trajectories, window_size=20, n_segments=5):
    segment_data = [{'X': [], 'y': []} for _ in range(n_segments)]
    for traj in test_trajectories:
        T_traj = len(traj['r1'])
        t_vals = np.arange(window_size, T_traj + 1)
        t_segments = np.array_split(t_vals, n_segments)
        for seg_idx, t_seg in enumerate(t_segments):
            for t in t_seg:
                window = np.stack([traj['r1'][t-window_size:t], traj['r2'][t-window_size:t]], axis=1)
                segment_data[seg_idx]['X'].append(window)
                segment_data[seg_idx]['y'].append(traj['error_state'][t-1])
    for seg_idx in range(n_segments):
        segment_data[seg_idx]['X'] = np.asarray(segment_data[seg_idx]['X'])
        segment_data[seg_idx]['y'] = np.asarray(segment_data[seg_idx]['y'])
    return segment_data

segments = collect_segmented_windows(test_trajectories, WINDOW_SIZE, 5)

adapt_ps = copy.deepcopy(adapt_pseudo)
adapt_hy = copy.deepcopy(adapt_hybrid)
adapt_ps.ema_grads = None; adapt_ps.update_count = 0
adapt_hy.ema_grads = None; adapt_hy.update_count = 0

for i, seg in enumerate(segments):
    X_s, y_s = seg['X'], seg['y']
    
    static_gru.eval()
    with torch.no_grad():
        sp = static_gru(torch.tensor(X_s, dtype=torch.float32)).argmax(dim=1).numpy()
    
    pp, _ = adapt_ps.predict_adaptive(X_s, y_true=None, reset_ema=False)
    hp, _ = adapt_hy.predict_adaptive(X_s, y_true=y_s, reset_ema=False, supervised_every=SUPERVISED_EVERY)
    
    print(f"   Seg {i+1}: Static={accuracy(y_s, sp):.3f}, "
          f"Pseudo={accuracy(y_s, pp):.3f}, Hybrid={accuracy(y_s, hp):.3f}")

# ── Supervision sweep ──
print("\n6. Supervision frequency sweep...")
for rate in [5, 10, 50]:
    m = copy.deepcopy(adapt_hybrid)
    p, _ = m.predict_adaptive(X_test, y_true=y_test, reset_ema=True, supervised_every=rate)
    print(f"   Every {rate}: {accuracy(y_test, p):.4f}")

print("\n" + "=" * 50)
print("✓ All Phase 4 smoke tests passed!")
print("=" * 50)
