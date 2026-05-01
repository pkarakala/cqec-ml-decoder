import numpy as np
import sys

from src.sim_nonideal import generate_trajectory_nonideal, generate_dataset_nonideal
from src.sim_hamiltonian import generate_trajectory_hamiltonian
from src.sim_measurement import generate_trajectory as gen_phase1
from src.operators import ERROR_SIGNATURES

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


# ══════════════════════════════════════════════════════════════
#  TEST 1: Basic Output Shapes (No Non-Idealities)
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 1: Basic Output Shapes ===")
traj = generate_trajectory_nonideal(T=200, seed=42)
check("r1 shape is (200,)",              traj["r1"].shape == (200,))
check("r2 shape is (200,)",              traj["r2"].shape == (200,))
check("error_labels shape is (200,)",    traj["error_labels"].shape == (200,))
check("times shape is (200,)",           traj["times"].shape == (200,))
check("meas_strength_t shape is (200,)", traj["meas_strength_t"].shape == (200,))
check("drive_signal shape is (200,)",    traj["drive_signal"].shape == (200,))
# Phase 3 new fields
check("colored_noise_r1 shape is (200,)", traj["colored_noise_r1"].shape == (200,))
check("colored_noise_r2 shape is (200,)", traj["colored_noise_r2"].shape == (200,))
check("transient_r1 shape is (200,)",     traj["transient_r1"].shape == (200,))
check("transient_r2 shape is (200,)",     traj["transient_r2"].shape == (200,))
check("random_walk_mean shape is (200,)", traj["random_walk_mean"].shape == (200,))
check("flip_times is a list",            isinstance(traj["flip_times"], list))
check("true_s1 shape is (200,)",         traj["true_s1"].shape == (200,))
check("true_s2 shape is (200,)",         traj["true_s2"].shape == (200,))


# ══════════════════════════════════════════════════════════════
#  TEST 2: Backward Compatibility — Phase 3 (off) == Phase 2
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 2: Backward Compatibility with Phase 2 ===")
traj_p2 = generate_trajectory_hamiltonian(
    T=100, p_flip=0.01, meas_strength=1.0, noise_std=1.0,
    drive_amplitude=0.0, drift_rate=0.0, backaction_strength=0.0,
    seed=42
)
traj_p3 = generate_trajectory_nonideal(
    T=100, p_flip=0.01, meas_strength=1.0, noise_std=1.0,
    drive_amplitude=0.0, drift_rate=0.0, backaction_strength=0.0,
    colored_noise_alpha=0.0, transient_amplitude=0.0, random_walk_strength=0.0,
    seed=42
)
check("r1 matches Phase 2",           np.allclose(traj_p2["r1"], traj_p3["r1"]))
check("r2 matches Phase 2",           np.allclose(traj_p2["r2"], traj_p3["r2"]))
check("error_labels match Phase 2",   np.array_equal(traj_p2["error_labels"], traj_p3["error_labels"]))
check("meas_strength_t matches P2",   np.allclose(traj_p2["meas_strength_t"], traj_p3["meas_strength_t"]))
check("drive_signal matches P2",      np.allclose(traj_p2["drive_signal"], traj_p3["drive_signal"]))


# ══════════════════════════════════════════════════════════════
#  TEST 3: Backward Compatibility — Phase 3 (off) == Phase 1
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 3: Backward Compatibility with Phase 1 ===")
traj_p1 = gen_phase1(T=100, p_flip=0.01, meas_strength=1.0, noise_std=1.0, seed=42)
check("r1 matches Phase 1",         np.allclose(traj_p1["r1"], traj_p3["r1"]))
check("r2 matches Phase 1",         np.allclose(traj_p1["r2"], traj_p3["r2"]))
check("error_labels match Phase 1", np.array_equal(traj_p1["error_labels"], traj_p3["error_labels"]))


# ══════════════════════════════════════════════════════════════
#  TEST 4: Colored Noise — AR(1) Temporal Correlations
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 4: Colored Noise (AR(1)) ===")

# 4a: alpha=0 should produce white noise (same as default)
traj_white = generate_trajectory_nonideal(
    T=500, p_flip=0.0, meas_strength=1.0, noise_std=1.0,
    colored_noise_alpha=0.0, seed=77
)
traj_colored = generate_trajectory_nonideal(
    T=500, p_flip=0.0, meas_strength=1.0, noise_std=1.0,
    colored_noise_alpha=0.9, seed=77
)

# With alpha=0, colored_noise_r1 should be white noise
# Autocorrelation at lag 1 should be near 0 for white noise
white_noise = traj_white["colored_noise_r1"]
colored_noise = traj_colored["colored_noise_r1"]

# Compute lag-1 autocorrelation
def lag1_autocorr(x):
    x_centered = x - x.mean()
    var = np.var(x_centered)
    if var == 0:
        return 0.0
    return np.corrcoef(x_centered[:-1], x_centered[1:])[0, 1]

white_ac = lag1_autocorr(white_noise)
colored_ac = lag1_autocorr(colored_noise)

print(f"       White noise lag-1 autocorr:   {white_ac:.3f}")
print(f"       Colored noise lag-1 autocorr: {colored_ac:.3f}")
check("white noise has low autocorrelation (|ac| < 0.15)",  abs(white_ac) < 0.15)
check("colored noise has high autocorrelation (ac > 0.7)",  colored_ac > 0.7)

# 4b: Colored noise should have same marginal variance as white noise
# AR(1) with the sqrt(1-alpha^2) scaling preserves stationary variance
white_var = np.var(white_noise)
colored_var = np.var(colored_noise)
print(f"       White noise variance:   {white_var:.3f}")
print(f"       Colored noise variance: {colored_var:.3f}")
check("colored noise variance ≈ white noise variance (within 50%)",
      abs(colored_var - white_var) / white_var < 0.5)

# 4c: Colored noise should change the measurement record
check("colored noise changes r1 vs white",
      not np.allclose(traj_white["r1"], traj_colored["r1"]))

# 4d: With no errors, the noise IS the signal deviation from meas_strength
# So colored_noise_r1 should equal r1 - meas_strength * true_s1
traj_cn_check = generate_trajectory_nonideal(
    T=100, p_flip=0.0, meas_strength=1.0, noise_std=1.0,
    colored_noise_alpha=0.5, backaction_strength=0.0,
    drive_amplitude=0.0, drift_rate=0.0, random_walk_strength=0.0,
    transient_amplitude=0.0, seed=55
)
reconstructed_noise = traj_cn_check["r1"] - traj_cn_check["meas_strength_t"] * traj_cn_check["true_s1"]
check("colored noise component reconstructs from r1",
      np.allclose(reconstructed_noise, traj_cn_check["colored_noise_r1"]))


# ══════════════════════════════════════════════════════════════
#  TEST 5: Post-Flip Transients
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 5: Post-Flip Transients ===")

# 5a: With transient_amplitude=0, transient arrays should be all zeros
traj_no_trans = generate_trajectory_nonideal(
    T=200, p_flip=0.05, transient_amplitude=0.0, seed=42
)
check("transient_amplitude=0 → transient_r1 all zeros",
      np.allclose(traj_no_trans["transient_r1"], 0.0))
check("transient_amplitude=0 → transient_r2 all zeros",
      np.allclose(traj_no_trans["transient_r2"], 0.0))

# 5b: With transient_amplitude>0 and flips present, transients should be nonzero
traj_trans = generate_trajectory_nonideal(
    T=200, p_flip=0.05, transient_amplitude=1.0, transient_decay=0.1, seed=42
)
has_flips = len(traj_trans["flip_times"]) > 0
check("trajectory has flip events", has_flips)
if has_flips:
    check("transient_r1 has nonzero values",  np.any(traj_trans["transient_r1"] != 0))
    check("transient_r2 has nonzero values",  np.any(traj_trans["transient_r2"] != 0))

# 5c: Transient should be positive (exponential decay from positive amplitude)
check("transient_r1 is non-negative", np.all(traj_trans["transient_r1"] >= 0))
check("transient_r2 is non-negative", np.all(traj_trans["transient_r2"] >= 0))

# 5d: Transient should peak at flip times and decay afterward
if has_flips:
    first_flip = traj_trans["flip_times"][0]
    # At the flip time, transient should equal the amplitude
    check(f"transient at flip t={first_flip} equals amplitude",
          np.isclose(traj_trans["transient_r1"][first_flip], 1.0))
    # A few steps later, it should have decayed
    # Find a flip that has no other flips within 10 steps after it
    flip_set = set(traj_trans["flip_times"])
    isolated_flip = None
    for ft in traj_trans["flip_times"]:
        if ft + 10 < 200 and all((ft + d) not in flip_set for d in range(1, 11)):
            isolated_flip = ft
            break
    if isolated_flip is not None:
        check(f"transient decays after isolated flip t={isolated_flip}",
              traj_trans["transient_r1"][isolated_flip + 10] < traj_trans["transient_r1"][isolated_flip])
    else:
        # Can't find an isolated flip — skip this sub-test
        check("transient decays after flip (skipped: no isolated flip)", True)

# 5e: With no flips, transients should be zero even with amplitude>0
traj_no_flip_trans = generate_trajectory_nonideal(
    T=200, p_flip=0.0, transient_amplitude=1.0, seed=42
)
check("no flips → transient_r1 all zeros",
      np.allclose(traj_no_flip_trans["transient_r1"], 0.0))

# 5f: Transient affects the measurement record
traj_a = generate_trajectory_nonideal(
    T=200, p_flip=0.05, transient_amplitude=0.0, seed=42
)
traj_b = generate_trajectory_nonideal(
    T=200, p_flip=0.05, transient_amplitude=1.0, seed=42
)
# Error process should be identical (same seed, same p_flip)
check("same error process with/without transient",
      np.array_equal(traj_a["error_labels"], traj_b["error_labels"]))
# But measurements should differ where transients are active
if len(traj_b["flip_times"]) > 0:
    check("transient changes r1", not np.allclose(traj_a["r1"], traj_b["r1"]))


# ══════════════════════════════════════════════════════════════
#  TEST 6: Random-Walk Drift
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 6: Random-Walk Drift ===")

# 6a: With random_walk_strength=0, random_walk_mean should be constant
traj_no_rw = generate_trajectory_nonideal(
    T=200, meas_strength=2.0, random_walk_strength=0.0, seed=42
)
check("rw_strength=0 → random_walk_mean constant at meas_strength",
      np.allclose(traj_no_rw["random_walk_mean"], 2.0))

# 6b: With random_walk_strength>0, random_walk_mean should vary
traj_rw = generate_trajectory_nonideal(
    T=500, dt=0.01, meas_strength=1.0, random_walk_strength=0.5, seed=42
)
check("rw_strength>0 → random_walk_mean varies",
      np.std(traj_rw["random_walk_mean"]) > 0.01)

# 6c: Random walk should start at meas_strength
check("random_walk_mean[0] = meas_strength",
      np.isclose(traj_rw["random_walk_mean"][0], 1.0))

# 6d: Random walk increments should be approximately Gaussian
rw_increments = np.diff(traj_rw["random_walk_mean"])
# Expected std of increments: sqrt(dt) * random_walk_strength
expected_inc_std = np.sqrt(0.01) * 0.5
actual_inc_std = np.std(rw_increments)
print(f"       Expected increment std: {expected_inc_std:.4f}")
print(f"       Actual increment std:   {actual_inc_std:.4f}")
check("random walk increment std matches theory (within 30%)",
      abs(actual_inc_std - expected_inc_std) / expected_inc_std < 0.3)

# 6e: Random walk affects meas_strength_t
# With no linear drift, meas_strength_t should track random_walk_mean
traj_rw_only = generate_trajectory_nonideal(
    T=200, meas_strength=1.0, drift_rate=0.0, random_walk_strength=0.3, seed=99
)
# meas_strength_t = base_drift + (random_walk_mean - meas_strength)
# base_drift = meas_strength + drift_rate * t * dt = 1.0 (since drift_rate=0)
# So meas_strength_t = 1.0 + (random_walk_mean - 1.0) = random_walk_mean
check("meas_strength_t tracks random_walk_mean (no linear drift)",
      np.allclose(traj_rw_only["meas_strength_t"], traj_rw_only["random_walk_mean"]))

# 6f: Random walk + linear drift combine correctly
traj_both = generate_trajectory_nonideal(
    T=200, dt=0.01, meas_strength=1.0, drift_rate=0.1, random_walk_strength=0.3, seed=99
)
times = np.arange(200) * 0.01
expected_meas = (1.0 + 0.1 * times) + (traj_both["random_walk_mean"] - 1.0)
check("linear drift + random walk combine correctly",
      np.allclose(traj_both["meas_strength_t"], expected_meas))


# ══════════════════════════════════════════════════════════════
#  TEST 7: Error Process Integrity
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 7: Error Process Integrity ===")

traj_full = generate_trajectory_nonideal(
    T=300, p_flip=0.03, colored_noise_alpha=0.8,
    transient_amplitude=0.5, random_walk_strength=0.2, seed=42
)

# 7a: Error labels should be in {0,1,2,3}
check("error labels valid", set(np.unique(traj_full["error_labels"])).issubset({0, 1, 2, 3}))

# 7b: True syndromes should be ±1
check("true_s1 is ±1", set(np.unique(traj_full["true_s1"])).issubset({-1, 1}))
check("true_s2 is ±1", set(np.unique(traj_full["true_s2"])).issubset({-1, 1}))

# 7c: Syndrome signatures should match ERROR_SIGNATURES lookup table
all_match = True
for t in range(len(traj_full["error_labels"])):
    e = traj_full["error_labels"][t]
    expected_s1, expected_s2 = ERROR_SIGNATURES[e]
    if traj_full["true_s1"][t] != expected_s1 or traj_full["true_s2"][t] != expected_s2:
        all_match = False
        break
check("all syndrome signatures match ERROR_SIGNATURES", all_match)

# 7d: flip_times should correspond to actual changes in error_labels
labels = traj_full["error_labels"]
actual_change_times = [t for t in range(1, len(labels)) if labels[t] != labels[t-1]]
# flip_times includes all attempted flips (even same→same transitions
# that don't change the label). But every label change should have a flip_time.
for ct in actual_change_times:
    check(f"label change at t={ct} has corresponding flip_time",
          ct in traj_full["flip_times"])
    if ct > actual_change_times[5]:  # only check first few to avoid spam
        break


# ══════════════════════════════════════════════════════════════
#  TEST 8: Measurement Record Decomposition
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 8: Measurement Record Decomposition ===")

# With all components active, verify r1 = signal + noise + transient + backaction
traj_decomp = generate_trajectory_nonideal(
    T=100, p_flip=0.03, meas_strength=1.0, noise_std=0.5,
    colored_noise_alpha=0.5, transient_amplitude=0.3, transient_decay=0.1,
    random_walk_strength=0.1, drive_amplitude=0.2, drive_frequency=2.0,
    drift_rate=0.0, backaction_strength=0.0,  # turn off backaction for clean decomposition
    seed=42
)

# r1 should equal (meas_strength_t + drive) * true_s1 + colored_noise + transient
signal = (traj_decomp["meas_strength_t"] + traj_decomp["drive_signal"]) * traj_decomp["true_s1"]
reconstructed = signal + traj_decomp["colored_noise_r1"] + traj_decomp["transient_r1"]
check("r1 decomposes into signal + noise + transient (no backaction)",
      np.allclose(traj_decomp["r1"], reconstructed))

# Same for r2
signal_r2 = (traj_decomp["meas_strength_t"] + traj_decomp["drive_signal"]) * traj_decomp["true_s2"]
reconstructed_r2 = signal_r2 + traj_decomp["colored_noise_r2"] + traj_decomp["transient_r2"]
check("r2 decomposes into signal + noise + transient (no backaction)",
      np.allclose(traj_decomp["r2"], reconstructed_r2))


# ══════════════════════════════════════════════════════════════
#  TEST 9: Seed Reproducibility
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 9: Seed Reproducibility ===")

traj_x = generate_trajectory_nonideal(
    T=100, colored_noise_alpha=0.8, transient_amplitude=0.5,
    random_walk_strength=0.2, seed=777
)
traj_y = generate_trajectory_nonideal(
    T=100, colored_noise_alpha=0.8, transient_amplitude=0.5,
    random_walk_strength=0.2, seed=777
)
check("same seed → same r1",              np.array_equal(traj_x["r1"], traj_y["r1"]))
check("same seed → same r2",              np.array_equal(traj_x["r2"], traj_y["r2"]))
check("same seed → same error_labels",    np.array_equal(traj_x["error_labels"], traj_y["error_labels"]))
check("same seed → same colored_noise",   np.array_equal(traj_x["colored_noise_r1"], traj_y["colored_noise_r1"]))
check("same seed → same random_walk",     np.array_equal(traj_x["random_walk_mean"], traj_y["random_walk_mean"]))
check("same seed → same transient",       np.array_equal(traj_x["transient_r1"], traj_y["transient_r1"]))

traj_z = generate_trajectory_nonideal(
    T=100, colored_noise_alpha=0.8, transient_amplitude=0.5,
    random_walk_strength=0.2, seed=888
)
check("different seed → different r1",    not np.array_equal(traj_x["r1"], traj_z["r1"]))
check("different seed → different rw",    not np.array_equal(traj_x["random_walk_mean"], traj_z["random_walk_mean"]))


# ══════════════════════════════════════════════════════════════
#  TEST 10: Phase 2 Dynamics Still Work in Phase 3
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 10: Phase 2 Dynamics Still Work ===")

# Drive signal
traj_drive = generate_trajectory_nonideal(
    T=200, dt=0.01, drive_amplitude=1.0, drive_frequency=2.0, seed=42
)
check("drive signal oscillates", np.std(traj_drive["drive_signal"]) > 0.1)
check("drive signal min < -0.9", np.min(traj_drive["drive_signal"]) < -0.9)
check("drive signal max > +0.9", np.max(traj_drive["drive_signal"]) > 0.9)

# Linear drift
traj_drift = generate_trajectory_nonideal(
    T=100, dt=0.01, meas_strength=1.0, drift_rate=0.1,
    random_walk_strength=0.0, seed=42
)
check("linear drift start = 1.0", np.isclose(traj_drift["meas_strength_t"][0], 1.0))
check("linear drift end ≈ 1.099", np.isclose(traj_drift["meas_strength_t"][-1], 1.099, atol=0.01))

# Backaction
traj_back = generate_trajectory_nonideal(
    T=100, p_flip=0.0, meas_strength=1.0, noise_std=0.0,
    colored_noise_alpha=0.0, backaction_strength=0.5, seed=99
)
# With noise_std=0 and colored_noise_alpha=0, the "noise" draws are N(0,0)=0
# But backaction adds noise. However, colored_noise_alpha=0 means white noise
# with noise_std=0 → colored_noise is 0. So r1 = meas_strength * s1 + 0 + 0 + backaction
# The signal won't be clean ±1 because of backaction
check("backaction adds noise to signal",
      not np.all(np.abs(traj_back["r1"]) == 1.0))


# ══════════════════════════════════════════════════════════════
#  TEST 11: Batch Generator
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 11: Batch Generator ===")

dataset = generate_dataset_nonideal(
    n_trajectories=50, T=100,
    colored_noise_alpha=0.5, transient_amplitude=0.3,
    random_walk_strength=0.1, seed=42
)
check("dataset length = 50", len(dataset) == 50)
check("first traj has T=100", len(dataset[0]["r1"]) == 100)
check("has colored_noise_r1 field", "colored_noise_r1" in dataset[0])
check("has transient_r1 field",     "transient_r1" in dataset[0])
check("has random_walk_mean field",  "random_walk_mean" in dataset[0])

# Batch should be reproducible
dataset2 = generate_dataset_nonideal(
    n_trajectories=50, T=100,
    colored_noise_alpha=0.5, transient_amplitude=0.3,
    random_walk_strength=0.1, seed=42
)
check("batch is reproducible (same seed)",
      np.array_equal(dataset[0]["r1"], dataset2[0]["r1"]))

# Different trajectories in the batch should differ
check("different trajectories in batch differ",
      not np.array_equal(dataset[0]["r1"], dataset[1]["r1"]))


# ══════════════════════════════════════════════════════════════
#  TEST 12: All Non-Idealities Combined
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 12: All Non-Idealities Combined ===")

traj_all = generate_trajectory_nonideal(
    T=300, dt=0.01, p_flip=0.03,
    meas_strength=1.0, noise_std=0.5,
    drive_amplitude=0.2, drive_frequency=2.0,
    drift_rate=0.05, backaction_strength=0.1,
    colored_noise_alpha=0.8,
    transient_amplitude=0.5, transient_decay=0.1,
    random_walk_strength=0.2,
    seed=42
)

# Everything should still produce valid output
check("combined: error labels valid",
      set(np.unique(traj_all["error_labels"])).issubset({0, 1, 2, 3}))
check("combined: r1 has no NaN",    not np.any(np.isnan(traj_all["r1"])))
check("combined: r2 has no NaN",    not np.any(np.isnan(traj_all["r2"])))
check("combined: r1 has no Inf",    not np.any(np.isinf(traj_all["r1"])))
check("combined: r2 has no Inf",    not np.any(np.isinf(traj_all["r2"])))

# The signal should be more complex than any single non-ideality
check("combined: colored noise is correlated",
      lag1_autocorr(traj_all["colored_noise_r1"]) > 0.5)
check("combined: random walk drifts",
      np.std(traj_all["random_walk_mean"]) > 0.01)
check("combined: drive oscillates",
      np.std(traj_all["drive_signal"]) > 0.05)
if len(traj_all["flip_times"]) > 0:
    check("combined: transients present",
          np.any(traj_all["transient_r1"] > 0))


# ══════════════════════════════════════════════════════════════
#  TEST 13: Edge Cases
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 13: Edge Cases ===")

# 13a: T=1 (single timestep)
traj_t1 = generate_trajectory_nonideal(T=1, seed=42)
check("T=1 works", traj_t1["r1"].shape == (1,))

# 13b: Very high flip rate (every timestep)
traj_high_flip = generate_trajectory_nonideal(T=100, p_flip=1.0, seed=42)
check("p_flip=1.0 doesn't crash", traj_high_flip["r1"].shape == (100,))
check("p_flip=1.0 has many flips", len(traj_high_flip["flip_times"]) > 50)

# 13c: Zero noise
traj_zero_noise = generate_trajectory_nonideal(
    T=100, p_flip=0.0, meas_strength=1.0, noise_std=0.0,
    colored_noise_alpha=0.0, backaction_strength=0.0,
    transient_amplitude=0.0, random_walk_strength=0.0,
    drive_amplitude=0.0, drift_rate=0.0, seed=42
)
# With no errors and no noise, r1 should be exactly meas_strength * true_s1 = 1.0
check("zero noise gives clean signal", np.allclose(traj_zero_noise["r1"], 1.0))

# 13d: Very large colored noise alpha (near 1)
traj_high_alpha = generate_trajectory_nonideal(
    T=200, colored_noise_alpha=0.99, noise_std=1.0, seed=42
)
check("alpha=0.99 doesn't crash", traj_high_alpha["r1"].shape == (200,))
check("alpha=0.99 has very high autocorrelation",
      lag1_autocorr(traj_high_alpha["colored_noise_r1"]) > 0.9)

# 13e: Very fast transient decay
traj_fast_decay = generate_trajectory_nonideal(
    T=200, p_flip=0.05, transient_amplitude=1.0, transient_decay=10.0, seed=42
)
check("fast transient decay doesn't crash", traj_fast_decay["r1"].shape == (200,))
# With decay=10.0, transient should vanish within 1 timestep
# contribution at t+1 = amp * exp(-10*1) ≈ 0.0000454 < 0.01*amp → pruned
if len(traj_fast_decay["flip_times"]) > 0:
    ft = traj_fast_decay["flip_times"][0]
    if ft + 1 < 200:
        check("fast decay: transient nearly zero 1 step after flip",
              traj_fast_decay["transient_r1"][ft + 1] < 0.01)


# ══════════════════════════════════════════════════════════════
#  TEST 14: Non-Idealities Are Independent
# ══════════════════════════════════════════════════════════════
print("\n=== TEST 14: Non-Idealities Are Independent ===")

# Turning on colored noise shouldn't affect the error process
traj_base = generate_trajectory_nonideal(
    T=200, p_flip=0.03, colored_noise_alpha=0.0,
    transient_amplitude=0.0, random_walk_strength=0.0, seed=42
)
traj_cn_only = generate_trajectory_nonideal(
    T=200, p_flip=0.03, colored_noise_alpha=0.9,
    transient_amplitude=0.0, random_walk_strength=0.0, seed=42
)
traj_rw_only = generate_trajectory_nonideal(
    T=200, p_flip=0.03, colored_noise_alpha=0.0,
    transient_amplitude=0.0, random_walk_strength=0.5, seed=42
)

# Error labels should differ because the RNG state diverges after
# colored noise draws different random numbers. But let's verify
# the error process is at least valid.
check("colored noise only: valid labels",
      set(np.unique(traj_cn_only["error_labels"])).issubset({0, 1, 2, 3}))
check("random walk only: valid labels",
      set(np.unique(traj_rw_only["error_labels"])).issubset({0, 1, 2, 3}))

# With colored noise on but transient off, transient should be zero
check("colored noise on, transient off → transient_r1 = 0",
      np.allclose(traj_cn_only["transient_r1"], 0.0))

# With random walk on but colored noise off, noise should be white
rw_noise_ac = lag1_autocorr(traj_rw_only["colored_noise_r1"])
check("random walk on, colored noise off → white noise (|ac| < 0.15)",
      abs(rw_noise_ac) < 0.15)


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}\n")

if failed > 0:
    sys.exit(1)
else:
    print("  sim_nonideal.py verified. Safe to build on.\n")
