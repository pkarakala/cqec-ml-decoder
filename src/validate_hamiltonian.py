import numpy as np
import sys

# Test the new Hamiltonian simulator
from src.sim_hamiltonian import generate_trajectory_hamiltonian, generate_dataset_hamiltonian

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

print("\n=== TEST 1: Basic Shapes (No Dynamics) ===")
traj = generate_trajectory_hamiltonian(T=200, seed=42)
check("r1 shape is (200,)",              traj["r1"].shape == (200,))
check("r2 shape is (200,)",              traj["r2"].shape == (200,))
check("error_labels shape is (200,)",    traj["error_labels"].shape == (200,))
check("times shape is (200,)",           traj["times"].shape == (200,))
check("meas_strength_t shape is (200,)", traj["meas_strength_t"].shape == (200,))
check("drive_signal shape is (200,)",    traj["drive_signal"].shape == (200,))

print("\n=== TEST 2: Drive Signal Properties ===")
# With drive_amplitude=0, drive_signal should be all zeros
traj_no_drive = generate_trajectory_hamiltonian(T=100, drive_amplitude=0.0, seed=42)
check("drive=0 → drive_signal all zeros", np.allclose(traj_no_drive["drive_signal"], 0.0))

# With drive_amplitude=1.0, drive_signal should oscillate between -1 and 1
traj_with_drive = generate_trajectory_hamiltonian(
    T=200, dt=0.01, drive_amplitude=1.0, drive_frequency=2.0, seed=42
)
check("drive signal min ≈ -1", np.min(traj_with_drive["drive_signal"]) < -0.9)
check("drive signal max ≈ +1", np.max(traj_with_drive["drive_signal"]) > 0.9)
check("drive signal mean ≈ 0", np.abs(np.mean(traj_with_drive["drive_signal"])) < 0.1)

print("\n=== TEST 3: Drift Properties ===")
# With drift_rate=0, meas_strength should be constant
traj_no_drift = generate_trajectory_hamiltonian(T=100, meas_strength=2.0, drift_rate=0.0, seed=42)
check("drift=0 → constant meas_strength", np.allclose(traj_no_drift["meas_strength_t"], 2.0))

# With drift_rate=0.1, meas_strength should increase linearly
traj_with_drift = generate_trajectory_hamiltonian(
    T=100, dt=0.01, meas_strength=1.0, drift_rate=0.1, seed=42
)
# At t=0: 1.0, at t=99*0.01=0.99: 1.0 + 0.1*0.99 = 1.099
check("drift start = 1.0",    np.isclose(traj_with_drift["meas_strength_t"][0],  1.0))
check("drift end ≈ 1.099",    np.isclose(traj_with_drift["meas_strength_t"][-1], 1.099, atol=0.01))
# Should be monotonically increasing
diffs = np.diff(traj_with_drift["meas_strength_t"])
check("drift is monotonic",   np.all(diffs >= 0))

print("\n=== TEST 4: Backaction Noise ===")
# With backaction_strength=0, measurements should only have readout noise
# Generate two trajectories with same seed but different backaction
traj_no_back = generate_trajectory_hamiltonian(
    T=100, p_flip=0.0, meas_strength=1.0, noise_std=0.0, backaction_strength=0.0, seed=99
)
traj_with_back = generate_trajectory_hamiltonian(
    T=100, p_flip=0.0, meas_strength=1.0, noise_std=0.0, backaction_strength=0.5, seed=99
)
# With no errors and no readout noise, backaction=0 should give clean ±1
check("backaction=0 gives clean signal", np.all(np.abs(traj_no_back["r1"]) == 1.0))
# With backaction>0, should have noise even with noise_std=0
check("backaction>0 adds noise", not np.all(np.abs(traj_with_back["r1"]) == 1.0))

print("\n=== TEST 5: Error Process Still Works ===")
# Error labels should still be in {0,1,2,3}
check("error labels valid", set(np.unique(traj["error_labels"])).issubset({0,1,2,3}))
# True syndromes should still be ±1
check("true_s1 is ±1", set(np.unique(traj["true_s1"])).issubset({-1, 1}))
check("true_s2 is ±1", set(np.unique(traj["true_s2"])).issubset({-1, 1}))

print("\n=== TEST 6: Syndrome Signature Consistency ===")
# Even with dynamics, error signatures should match ERROR_SIGNATURES
from src.operators import ERROR_SIGNATURES
for t in range(len(traj["error_labels"])):
    e = traj["error_labels"][t]
    expected_s1, expected_s2 = ERROR_SIGNATURES[e]
    check(f"t={t}: signature match", 
          traj["true_s1"][t] == expected_s1 and traj["true_s2"][t] == expected_s2)
    if t >= 20:  # Only check first 20 to avoid spam
        break

print("\n=== TEST 7: Measurement Records Are Noisy ===")
# With noise_std>0, r1 and r2 should NOT equal true_s1 and true_s2
check("r1 has noise", not np.allclose(traj["r1"], traj["true_s1"]))
check("r2 has noise", not np.allclose(traj["r2"], traj["true_s2"]))

print("\n=== TEST 8: Drive Affects Measurement ===")
# Generate two trajectories with same errors but different drive
np.random.seed(123)
traj_a = generate_trajectory_hamiltonian(
    T=100, p_flip=0.0, drive_amplitude=0.0, seed=123
)
traj_b = generate_trajectory_hamiltonian(
    T=100, p_flip=0.0, drive_amplitude=0.5, seed=123
)
# Error labels should be identical (same error process)
check("same errors with/without drive", np.array_equal(traj_a["error_labels"], traj_b["error_labels"]))
# But r1 and r2 should differ (drive affects measurement)
check("drive changes r1", not np.allclose(traj_a["r1"], traj_b["r1"]))
check("drive changes r2", not np.allclose(traj_a["r2"], traj_b["r2"]))

print("\n=== TEST 9: Seed Reproducibility ===")
traj_x = generate_trajectory_hamiltonian(T=100, seed=777)
traj_y = generate_trajectory_hamiltonian(T=100, seed=777)
check("same seed → same r1",           np.array_equal(traj_x["r1"], traj_y["r1"]))
check("same seed → same error_labels", np.array_equal(traj_x["error_labels"], traj_y["error_labels"]))

traj_z = generate_trajectory_hamiltonian(T=100, seed=888)
check("different seed → different r1", not np.array_equal(traj_x["r1"], traj_z["r1"]))

print("\n=== TEST 10: Batch Generator ===")
dataset = generate_dataset_hamiltonian(
    n_trajectories=50, T=100, drive_amplitude=0.2, drift_rate=0.05, seed=42
)
check("dataset length = 50", len(dataset) == 50)
check("first traj has T=100",  len(dataset[0]["r1"]) == 100)
check("has drive_signal field", "drive_signal" in dataset[0])
check("has meas_strength_t field", "meas_strength_t" in dataset[0])

print("\n=== TEST 11: Dynamics Combine Correctly ===")
# Generate trajectory with ALL dynamics turned on
traj_full = generate_trajectory_hamiltonian(
    T=200, dt=0.01,
    meas_strength=1.0,
    noise_std=0.5,
    drive_amplitude=0.3,
    drive_frequency=2.0,
    drift_rate=0.05,
    backaction_strength=0.1,
    seed=42
)
# meas_strength_t should drift
check("drift present in meas_strength_t", 
      not np.allclose(traj_full["meas_strength_t"], traj_full["meas_strength_t"][0]))
# drive_signal should oscillate
check("drive present in drive_signal",
      np.std(traj_full["drive_signal"]) > 0.1)
# Measurements should be noisier than just syndrome
check("combined dynamics create complex signal",
      np.std(traj_full["r1"]) > 0.5)

print("\n=== TEST 12: Compatibility with Phase 1 (No Dynamics) ===")
from src.sim_measurement import generate_trajectory as gen_phase1

traj_p1 = gen_phase1(T=100, p_flip=0.01, meas_strength=1.0, noise_std=1.0, seed=42)
traj_p2 = generate_trajectory_hamiltonian(
    T=100, p_flip=0.01, meas_strength=1.0, noise_std=1.0,
    drive_amplitude=0.0, drift_rate=0.0, backaction_strength=0.0,
    seed=42
)
check("Phase 2 (no dynamics) matches Phase 1 r1", np.allclose(traj_p1["r1"], traj_p2["r1"]))
check("Phase 2 (no dynamics) matches Phase 1 r2", np.allclose(traj_p1["r2"], traj_p2["r2"]))
check("Phase 2 (no dynamics) matches Phase 1 labels", 
      np.array_equal(traj_p1["error_labels"], traj_p2["error_labels"]))

# ─── SUMMARY ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}\n")

if failed > 0:
    sys.exit(1)
else:
    print("  sim_hamiltonian.py verified. Safe to build on.\n")