import numpy as np
from .operators import ERROR_SIGNATURES


# ─── Non-Ideal Continuous Measurement Simulator ──────────────
# Phase 3 upgrade: adds three realistic non-idealities from
# Convy et al. that break the Bayesian filter's assumptions.
#
# New non-idealities:
# 1. Colored noise (AR(1) / Ornstein-Uhlenbeck process)
# 2. Post-flip transients (impulse response after error events)
# 3. Random-walk drift (Brownian motion in measurement means)
#
# These are the kinds of effects that appear in real quantum
# hardware but are not captured by idealized models. The Bayesian
# filter assumes white noise and static parameters — Phase 3 breaks
# both assumptions systematically.
# ───────────────────────────────────────────────────────────────


def generate_trajectory_nonideal(
    T: int = 200,
    dt: float = 0.01,
    p_flip: float = 0.01,
    meas_strength: float = 1.0,
    noise_std: float = 1.0,
    # Phase 2 parameters (still available)
    drive_amplitude: float = 0.0,
    drive_frequency: float = 1.0,
    drift_rate: float = 0.0,
    backaction_strength: float = 0.0,
    # Phase 3 NEW: colored noise
    colored_noise_alpha: float = 0.0,      # AR(1) correlation: 0=white, 0.9=highly colored
    # Phase 3 NEW: post-flip transient
    transient_amplitude: float = 0.0,      # strength of impulse after flip
    transient_decay: float = 0.1,          # decay rate (1/timesteps)
    # Phase 3 NEW: random-walk drift
    random_walk_strength: float = 0.0,     # diffusion coefficient for Brownian drift
    logical_state: int = 0,
    seed: int | None = None
) -> dict:
    """
    Generate one trajectory with Phase 3 non-idealities.

    Non-ideality #1: Colored noise (AR(1) process)
    ----------------------------------------------
    Instead of white Gaussian noise at each timestep, we use an
    autoregressive process:
        noise[t] = alpha * noise[t-1] + sqrt(1 - alpha^2) * white_noise[t]

    This creates temporal correlations in the noise. alpha=0 is white
    noise (Phase 2), alpha~0.9 is strongly colored (adjacent timesteps
    are correlated).

    Non-ideality #2: Post-flip transient
    ------------------------------------
    After an error occurs at time t_flip, the measurement channel
    experiences a short impulse/relaxation artifact:
        transient[t] = amplitude * exp(-decay * (t - t_flip))

    This mimics the physical effect where a bit flip causes a brief
    disturbance in the readout resonator that takes several timesteps
    to settle. The Bayesian filter has no model for this and gets confused.

    Non-ideality #3: Random-walk drift
    -----------------------------------
    The measurement mean undergoes Brownian motion:
        mean[t+1] = mean[t] + sqrt(dt) * random_walk_strength * randn()

    This is more realistic than linear drift (Phase 2) because real
    hardware has stochastic parameter fluctuations, not deterministic ones.

    Returns same dict as Phase 2 but with additional fields:
        - colored_noise_r1, colored_noise_r2: the AR(1) noise components
        - transient_r1, transient_r2: the post-flip transient signals
        - random_walk_mean: the time-varying Brownian mean
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

    # Phase 2 dynamics storage
    meas_strength_t  = np.zeros(T)
    drive_signal     = np.zeros(T)

    # Phase 3 NEW: colored noise storage
    colored_noise_r1 = np.zeros(T)
    colored_noise_r2 = np.zeros(T)
    colored_noise_r1_prev = 0.0  # AR(1) state
    colored_noise_r2_prev = 0.0

    # Phase 3 NEW: transient storage
    transient_r1     = np.zeros(T)
    transient_r2     = np.zeros(T)
    transient_queue_r1 = []  # list of (start_time, amplitude) for active transients
    transient_queue_r2 = []

    # Phase 3 NEW: random-walk drift storage
    random_walk_mean = np.zeros(T)
    random_walk_mean[0] = meas_strength  # initialize at base meas_strength

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

            # Phase 3 NEW: trigger post-flip transient
            if transient_amplitude > 0:
                # Add a transient impulse to both channels
                transient_queue_r1.append((t, transient_amplitude))
                transient_queue_r2.append((t, transient_amplitude))

        # ─── Step 2: Look up true syndrome ────────────────────
        s1_true, s2_true = ERROR_SIGNATURES[current_error]
        true_s1[t] = s1_true
        true_s2[t] = s2_true
        error_labels[t] = current_error

        # ─── Step 3a: Phase 2 time-dependent effects ──────────
        # Linear drift (Phase 2)
        base_drift = meas_strength + drift_rate * t * dt

        # Phase 3 NEW: Random-walk drift (Brownian motion)
        if t > 0 and random_walk_strength > 0:
            random_walk_mean[t] = random_walk_mean[t-1] + np.sqrt(dt) * random_walk_strength * rng.normal()
        elif t > 0:
            random_walk_mean[t] = random_walk_mean[t-1]

        # Combined measurement strength: base + linear drift + random walk
        meas_strength_t[t] = base_drift + (random_walk_mean[t] - meas_strength)

        # Coherent drive (Phase 2)
        drive_signal[t] = drive_amplitude * np.cos(drive_frequency * t * dt)

        # ─── Step 3b: Phase 3 colored noise (AR(1)) ───────────
        if colored_noise_alpha > 0:
            # AR(1) process: noise[t] = alpha * noise[t-1] + sqrt(1-alpha^2) * white_noise
            white_r1 = rng.normal(0, noise_std)
            white_r2 = rng.normal(0, noise_std)
            colored_noise_r1[t] = (colored_noise_alpha * colored_noise_r1_prev
                                   + np.sqrt(1 - colored_noise_alpha**2) * white_r1)
            colored_noise_r2[t] = (colored_noise_alpha * colored_noise_r2_prev
                                   + np.sqrt(1 - colored_noise_alpha**2) * white_r2)
            colored_noise_r1_prev = colored_noise_r1[t]
            colored_noise_r2_prev = colored_noise_r2[t]
        else:
            # White noise (Phase 2 default)
            colored_noise_r1[t] = rng.normal(0, noise_std)
            colored_noise_r2[t] = rng.normal(0, noise_std)

        # ─── Step 3c: Phase 3 post-flip transient ─────────────
        # Sum all active transients (exponential decay from each flip event)
        transient_r1[t] = 0.0
        transient_r2[t] = 0.0

        # Process r1 transients
        active_r1 = []
        for (t_start, amp) in transient_queue_r1:
            time_since_flip = t - t_start
            contribution = amp * np.exp(-transient_decay * time_since_flip)
            transient_r1[t] += contribution
            # Keep in queue if still significant
            if contribution > 0.01 * amp:
                active_r1.append((t_start, amp))
        transient_queue_r1 = active_r1

        # Process r2 transients
        active_r2 = []
        for (t_start, amp) in transient_queue_r2:
            time_since_flip = t - t_start
            contribution = amp * np.exp(-transient_decay * time_since_flip)
            transient_r2[t] += contribution
            if contribution > 0.01 * amp:
                active_r2.append((t_start, amp))
        transient_queue_r2 = active_r2

        # ─── Step 4: Backaction noise (Phase 2) ───────────────
        if backaction_strength > 0:
            backaction_noise_1 = rng.normal(0, backaction_strength)
            backaction_noise_2 = rng.normal(0, backaction_strength)
        else:
            backaction_noise_1 = 0.0
            backaction_noise_2 = 0.0

        # ─── Step 5: Assemble full measurement record ─────────
        # Formula: r(t) = [meas_strength(t) + drive(t)] * syndrome 
        #                 + colored_noise + transient + backaction
        r1[t] = ((meas_strength_t[t] + drive_signal[t]) * s1_true
                 + colored_noise_r1[t] + transient_r1[t] + backaction_noise_1)
        r2[t] = ((meas_strength_t[t] + drive_signal[t]) * s2_true
                 + colored_noise_r2[t] + transient_r2[t] + backaction_noise_2)

    return {
        "times":           times,
        "r1":              r1,
        "r2":              r2,
        "true_s1":         true_s1,
        "true_s2":         true_s2,
        "error_labels":    error_labels,
        "flip_times":      flip_times,
        # Phase 2 fields
        "meas_strength_t": meas_strength_t,
        "drive_signal":    drive_signal,
        # Phase 3 NEW fields
        "colored_noise_r1": colored_noise_r1,
        "colored_noise_r2": colored_noise_r2,
        "transient_r1":     transient_r1,
        "transient_r2":     transient_r2,
        "random_walk_mean": random_walk_mean,
    }


# ─── Batch Generator ──────────────────────────────────────────
def generate_dataset_nonideal(
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
    colored_noise_alpha: float = 0.0,
    transient_amplitude: float = 0.0,
    transient_decay: float = 0.1,
    random_walk_strength: float = 0.0,
    seed: int | None = None
) -> list[dict]:
    """
    Generate a batch of trajectories with Phase 3 non-idealities.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_trajectories)

    dataset = []
    for i in range(n_trajectories):
        logical_state = rng.integers(0, 2)
        traj = generate_trajectory_nonideal(
            T=T,
            dt=dt,
            p_flip=p_flip,
            meas_strength=meas_strength,
            noise_std=noise_std,
            drive_amplitude=drive_amplitude,
            drive_frequency=drive_frequency,
            drift_rate=drift_rate,
            backaction_strength=backaction_strength,
            colored_noise_alpha=colored_noise_alpha,
            transient_amplitude=transient_amplitude,
            transient_decay=transient_decay,
            random_walk_strength=random_walk_strength,
            logical_state=logical_state,
            seed=int(seeds[i])
        )
        dataset.append(traj)

    return dataset


# ─── Compatibility Check ──────────────────────────────────────
def test_compatibility():
    """
    Verify that Phase 3 with all non-idealities turned off
    produces the same output as Phase 2.
    """
    from .sim_hamiltonian import generate_trajectory_hamiltonian

    # Generate with Phase 2 and Phase 3 (non-idealities off)
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

    # Should have identical measurements
    assert np.allclose(traj_p2["r1"], traj_p3["r1"]), "r1 mismatch"
    assert np.allclose(traj_p2["r2"], traj_p3["r2"]), "r2 mismatch"
    assert np.array_equal(traj_p2["error_labels"], traj_p3["error_labels"]), "labels mismatch"

    print("✓ Compatibility check passed: Phase 3 with non-idealities=0 matches Phase 2")


if __name__ == "__main__":
    test_compatibility()
