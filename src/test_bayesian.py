import numpy as np
import sys

from src.bayesian_filter import BayesianFilter
from src.sim_measurement import generate_trajectory
from src.sim_hamiltonian import generate_trajectory_hamiltonian
from src.datasets import create_windows

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}")
        failed += 1

print("\n=== TEST 1: Initialization ===")
bf = BayesianFilter(p_flip=0.01, meas_strength=1.0, noise_std=1.0)
check("filter has 4 states", bf.n_states == 4)
check("transition matrix is 4x4", bf.transition_matrix.shape == (4, 4))
check("syndromes has 4 entries", len(bf.syndromes) == 4)

print("\n=== TEST 2: Transition Matrix Properties ===")
# Each row should sum to 1 (it's a probability distribution)
row_sums = bf.transition_matrix.sum(axis=1)
check("all rows sum to 1", np.allclose(row_sums, 1.0))

# Diagonal should have high probability (stay in current state)
diag = np.diag(bf.transition_matrix)
check("diagonal entries are highest in each row", 
      all(diag[i] == max(bf.transition_matrix[i, :]) for i in range(4)))

# Transition probabilities should be non-negative
check("all entries non-negative", np.all(bf.transition_matrix >= 0))

print("\n=== TEST 3: Observation Likelihood ===")
# State 0 has syndrome (+1, +1)
# If we measure (1.0, 1.0) with meas_strength=1.0 and low noise,
# likelihood should be highest for state 0
bf_test = BayesianFilter(p_flip=0.01, meas_strength=1.0, noise_std=0.1)

likelihoods = [bf_test.observation_likelihood(1.0, 1.0, s) for s in range(4)]
check("state 0 has highest likelihood for (1,1)", np.argmax(likelihoods) == 0)

# State 1 has syndrome (-1, +1)
likelihoods = [bf_test.observation_likelihood(-1.0, 1.0, s) for s in range(4)]
check("state 1 has highest likelihood for (-1,+1)", np.argmax(likelihoods) == 1)

# State 2 has syndrome (-1, -1)
likelihoods = [bf_test.observation_likelihood(-1.0, -1.0, s) for s in range(4)]
check("state 2 has highest likelihood for (-1,-1)", np.argmax(likelihoods) == 2)

# State 3 has syndrome (+1, -1)
likelihoods = [bf_test.observation_likelihood(1.0, -1.0, s) for s in range(4)]
check("state 3 has highest likelihood for (+1,-1)", np.argmax(likelihoods) == 3)

print("\n=== TEST 4: Prediction on Clean Data ===")
# Create a fake window with clean measurements (no noise)
# Window where syndrome is clearly (+1, +1) → should predict state 0
X_clean = np.zeros((1, 20, 2))
X_clean[0, :, 0] = 1.0  # r1 = +1
X_clean[0, :, 1] = 1.0  # r2 = +1

pred = bf_test.predict(X_clean)
check("clean (+1,+1) predicts state 0", pred[0] == 0)

# Window where syndrome is clearly (-1, +1) → should predict state 1
X_clean[0, :, 0] = -1.0
X_clean[0, :, 1] = 1.0
pred = bf_test.predict(X_clean)
check("clean (-1,+1) predicts state 1", pred[0] == 1)

print("\n=== TEST 5: Prediction Shape ===")
# Generate fake batch of windows
X_batch = np.random.randn(100, 20, 2)
preds = bf.predict(X_batch)
check("output shape is (100,)", preds.shape == (100,))
check("all predictions in {0,1,2,3}", set(np.unique(preds)).issubset({0,1,2,3}))

print("\n=== TEST 6: Performance on Phase 1 Data ===")
traj = generate_trajectory(T=300, p_flip=0.02, meas_strength=1.0, noise_std=1.0, seed=42)
windowed = create_windows(traj, window_size=20)

bf_p1 = BayesianFilter(p_flip=0.02, meas_strength=1.0, noise_std=1.0)
preds = bf_p1.predict(windowed["X"])
accuracy = (preds == windowed["y"]).mean()

print(f"       Phase 1 accuracy: {accuracy:.4f}")
check("Phase 1 accuracy > 0.85", accuracy > 0.85)

print("\n=== TEST 7: Comparison with Threshold Decoder ===")
from src.decoders import ThresholdDecoder

td = ThresholdDecoder()
td_preds = td.predict(windowed["X"])
td_accuracy = (td_preds == windowed["y"]).mean()

print(f"       Threshold accuracy: {td_accuracy:.4f}")
print(f"       Bayesian accuracy:  {accuracy:.4f}")
check("Bayesian beats Threshold", accuracy >= td_accuracy)

print("\n=== TEST 8: Performance on Phase 2 Data (No Dynamics) ===")
# With dynamics turned off, should still work well
traj_p2_static = generate_trajectory_hamiltonian(
    T=300, p_flip=0.02, meas_strength=1.0, noise_std=1.0,
    drive_amplitude=0.0, drift_rate=0.0, backaction_strength=0.0,
    seed=42
)
windowed_p2 = create_windows(traj_p2_static, window_size=20)
preds_p2 = bf_p1.predict(windowed_p2["X"])
accuracy_p2 = (preds_p2 == windowed_p2["y"]).mean()

print(f"       Phase 2 (no dynamics) accuracy: {accuracy_p2:.4f}")
check("Phase 2 static accuracy > 0.85", accuracy_p2 > 0.85)

print("\n=== TEST 9: Performance Degrades with Dynamics ===")
# With drive and drift, Bayesian filter should do worse
# because its model assumptions are violated
traj_p2_dynamic = generate_trajectory_hamiltonian(
    T=300, p_flip=0.02, meas_strength=1.0, noise_std=1.0,
    drive_amplitude=0.3, drift_rate=0.1, backaction_strength=0.0,
    seed=42
)
windowed_dyn = create_windows(traj_p2_dynamic, window_size=20)
preds_dyn = bf_p1.predict(windowed_dyn["X"])
accuracy_dyn = (preds_dyn == windowed_dyn["y"]).mean()

print(f"       Phase 2 (with dynamics) accuracy: {accuracy_dyn:.4f}")
check("dynamics hurt Bayesian performance", accuracy_dyn < accuracy_p2)

print("\n=== TEST 10: Filter with Mismatched Parameters ===")
# If filter is initialized with wrong noise_std, performance should degrade
bf_wrong = BayesianFilter(p_flip=0.02, meas_strength=1.0, noise_std=0.1)  # too low
preds_wrong = bf_wrong.predict(windowed["X"])
accuracy_wrong = (preds_wrong == windowed["y"]).mean()

print(f"       Correct params accuracy: {accuracy:.4f}")
print(f"       Wrong params accuracy:   {accuracy_wrong:.4f}")
check("wrong noise_std hurts performance", accuracy_wrong < accuracy)

print("\n=== TEST 11: Filter Handles Edge Cases ===")
# Very noisy data
traj_noisy = generate_trajectory(T=200, p_flip=0.02, meas_strength=1.0, noise_std=5.0, seed=99)
windowed_noisy = create_windows(traj_noisy, window_size=20)
bf_noisy = BayesianFilter(p_flip=0.02, meas_strength=1.0, noise_std=5.0)
preds_noisy = bf_noisy.predict(windowed_noisy["X"])

check("handles very noisy data without crashing", preds_noisy.shape[0] > 0)
check("predictions still valid", set(np.unique(preds_noisy)).issubset({0,1,2,3}))

print("\n=== TEST 12: Deterministic Output ===")
# Same input should give same output
preds_a = bf.predict(windowed["X"][:10])
preds_b = bf.predict(windowed["X"][:10])
check("filter is deterministic", np.array_equal(preds_a, preds_b))

# ─── SUMMARY ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}\n")

if failed > 0:
    sys.exit(1)
else:
    print("  bayesian_filter.py verified. Safe to build on.\n")