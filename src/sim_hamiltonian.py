import numpy as np
from .operators import S1, S2, ERRORS, ERROR_SIGNATURES, ket_0L, ket_1L


# ─── Time-Dependent Hamiltonian Simulator ────────────────────
# Phase 2 upgrade: instead of just random bit flips with static
# syndromes, we now have continuous Hamiltonian evolution that
# causes the measurement signal to have structured dynamics even
# without errors present.
#
# Key additions:
# 1. Rotating drive — coherent oscillations in the qubits
# 2. Measurement drift — the signal strength slowly changes
# 3. Measurement backaction — the act of measuring perturbs the state
# ───────────────────────────────────────────────────────────────


def generate_trajectory_hamiltonian(
    T: int = 200,
    dt: float = 0.01,
    p_flip: float = 0.01,
    meas_strength: float = 1.0,
    noise_std: float = 1.0,
    drive_amplitude: float = 0.0,      # NEW: strength of rotating drive
    drive_frequency: float = 1.0,      # NEW: oscillation frequency (rad/timestep)
    drift_rate: float = 0.0,           # NEW: linear drift in meas_strength per timestep
    backaction_strength: float = 0.0,  # NEW: measurement-induced dephasing
    logical_state: int = 0,
    seed: int | None = None
) -> dict:
    """
    Generate one trajectory with time-dependent Hamiltonian dynamics.
    
    New physics compared to Phase 1:
    - drive_amplitude > 0: adds a coherent oscillation to the syndromes
      even when no error is present. Mimics a rotating drive field.
    - drift_rate > 0: the effective measurement strength changes linearly
      over time, simulating calibration drift.
    - backaction_strength > 0: measurement induces random phase kicks,
      adding extra noise beyond the readout noise.
    
    The measurement model becomes:
        r(t) = [meas_strength(t) + drive(t)] * true_syndrome + noise + backaction
    
    Returns same dict as Phase 1 but with additional fields:
        - meas_strength_t: array of shape (T,) — effective meas strength at each step
        - drive_signal:    array of shape (T,) — the drive contribution
    """
    rng = np.random.default_rng(seed)
    
    # ─── Initialize state tracking ────────────────────────────
    current_error = 0
    
    # Storage
    times            = np.arange(T) * dt
    r1               = np.zeros(T)
    r2               = np.zeros(T)
    true_s1          = np.zeros(T)
    true_s2          = np.zeros(T)
    error_labels     = np.zeros(T, dtype=int)
    flip_times       = []
    meas_strength_t  = np.zeros(T)  # time-dependent measurement strength
    drive_signal     = np.zeros(T)  # coherent drive contribution
    
    for t in range(T):
        # ─── Step 1: Maybe inject a bit-flip ──────────────────
        if rng.random() < p_flip:
            flipped_qubit = rng.integers(1, 4)
            if current_error == 0:
                current_error = flipped_qubit
            elif current_error == flipped_qubit:
                current_error = 0
            else:
                current_error = flipped_qubit
            flip_times.append(t)
        
        # ─── Step 2: Look up true syndrome ────────────────────
        s1_true, s2_true = ERROR_SIGNATURES[current_error]
        true_s1[t] = s1_true
        true_s2[t] = s2_true
        error_labels[t] = current_error
        
        # ─── Step 3: Time-dependent effects ───────────────────
        # Measurement strength drifts linearly
        meas_strength_t[t] = meas_strength + drift_rate * t * dt
        
        # Coherent drive adds a sinusoidal modulation
        # This affects the syndrome signal even when no error is present
        drive_signal[t] = drive_amplitude * np.cos(drive_frequency * t * dt)
        
        # ─── Step 4: Generate noisy measurement with dynamics ─
        # Base signal: (meas_strength + drive) * true_syndrome
        # Then add readout noise + backaction noise
        
        # Readout noise (Gaussian, as before)
        readout_noise_1 = rng.normal(0, noise_std)
        readout_noise_2 = rng.normal(0, noise_std)
        
        # Backaction noise (only draw if backaction_strength > 0)
        if backaction_strength > 0:
            backaction_noise_1 = rng.normal(0, backaction_strength)
            backaction_noise_2 = rng.normal(0, backaction_strength)
        else:
            backaction_noise_1 = 0.0
            backaction_noise_2 = 0.0
        
        # Full measurement record
        r1[t] = (meas_strength_t[t] + drive_signal[t]) * s1_true + readout_noise_1 + backaction_noise_1
        r2[t] = (meas_strength_t[t] + drive_signal[t]) * s2_true + readout_noise_2 + backaction_noise_2
        
    return {
        "times":           times,
        "r1":              r1,
        "r2":              r2,
        "true_s1":         true_s1,
        "true_s2":         true_s2,
        "error_labels":    error_labels,
        "flip_times":      flip_times,
        "meas_strength_t": meas_strength_t,  # NEW
        "drive_signal":    drive_signal,     # NEW
    }


# ─── Batch Generator ──────────────────────────────────────────
def generate_dataset_hamiltonian(
    n_trajectories: int = 1000,
    T: int = 200,
    dt: float = 0.01,
    p_flip: float = 0.01,
    meas_strength: float = 1.0,
    noise_std: float = 1.0,
    drive_amplitude: float = 0.0,
    drive_frequency: float = 1.0,
    drift_rate: float = 0.0,
    backaction_strength: float = 0.0,
    seed: int | None = None
) -> list[dict]:
    """
    Generate a batch of trajectories with Hamiltonian dynamics.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_trajectories)
    
    dataset = []
    for i in range(n_trajectories):
        logical_state = rng.integers(0, 2)
        traj = generate_trajectory_hamiltonian(
            T=T,
            dt=dt,
            p_flip=p_flip,
            meas_strength=meas_strength,
            noise_std=noise_std,
            drive_amplitude=drive_amplitude,
            drive_frequency=drive_frequency,
            drift_rate=drift_rate,
            backaction_strength=backaction_strength,
            logical_state=logical_state,
            seed=int(seeds[i])
        )
        dataset.append(traj)
    
    return dataset


# ─── Compatibility Check ──────────────────────────────────────
# Quick test: Phase 2 with all dynamics turned off should match Phase 1

def test_compatibility():
    """
    Verify that Phase 2 simulator with dynamics=0 produces the same
    output structure as Phase 1.
    """
    from .sim_measurement import generate_trajectory as gen_phase1
    
    # Generate with both simulators using same seed and no dynamics
    traj_p1 = gen_phase1(T=100, p_flip=0.01, meas_strength=1.0, noise_std=1.0, seed=42)
    traj_p2 = generate_trajectory_hamiltonian(
        T=100, p_flip=0.01, meas_strength=1.0, noise_std=1.0,
        drive_amplitude=0.0, drift_rate=0.0, backaction_strength=0.0,
        seed=42
    )
    
    # Both should have identical r1, r2, error_labels
    assert np.allclose(traj_p1["r1"], traj_p2["r1"]), "r1 mismatch"
    assert np.allclose(traj_p1["r2"], traj_p2["r2"]), "r2 mismatch"
    assert np.array_equal(traj_p1["error_labels"], traj_p2["error_labels"]), "labels mismatch"
    
    print("✓ Compatibility check passed: Phase 2 with dynamics=0 matches Phase 1")


if __name__ == "__main__":
    test_compatibility()