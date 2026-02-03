import numpy as np
from .operators import ERROR_SIGNATURES


# ─── Continuous Measurement Record Generator ─────────────────
# Simulates a single trajectory of noisy analog syndrome readout.
#
# The idea: at each timestep, the "true" stabilizer value is ±1.
# What you actually measure is that value + Gaussian noise.
# That's your continuous measurement record.
# ───────────────────────────────────────────────────────────────

def generate_trajectory(
    T: int = 200,           # number of timesteps
    dt: float = 0.01,       # time between steps
    p_flip: float = 0.01,   # probability of a bit-flip per timestep
    meas_strength: float = 1.0,  # how strong the measurement signal is (SNR)
    noise_std: float = 1.0,      # Gaussian noise on the readout
    logical_state: int = 0,      # 0 for |0_L>, 1 for |1_L>
    seed: int | None = None      # random seed for reproducibility
) -> dict:
    """
    Generate one full trajectory of continuous syndrome measurements
    with random bit-flip errors injected along the way.

    Returns a dict with:
        - times:        array of shape (T,)  — the time axis
        - r1:           array of shape (T,)  — noisy measurement record for S1
        - r2:           array of shape (T,)  — noisy measurement record for S2
        - true_s1:      array of shape (T,)  — the actual S1 value (±1) at each step
        - true_s2:      array of shape (T,)  — the actual S2 value (±1) at each step
        - error_labels: array of shape (T,)  — which error is active (0=none,1,2,3)
        - flip_times:   list of ints         — timesteps where a flip happened
    """
    rng = np.random.default_rng(seed)

    # ─── Initialize state tracking ────────────────────────────
    # current_error tracks which qubit (if any) is currently flipped
    # starts at 0 = no error
    current_error = 0

    # Storage arrays
    times       = np.arange(T) * dt
    r1          = np.zeros(T)
    r2          = np.zeros(T)
    true_s1     = np.zeros(T)
    true_s2     = np.zeros(T)
    error_labels = np.zeros(T, dtype=int)
    flip_times  = []

    for t in range(T):
        # ─── Step 1: Maybe inject a bit-flip error ──────────────
        # Each timestep, with probability p_flip, one of the 3 qubits
        # flips. If one is already flipped and the SAME qubit flips
        # again, it cancels out (back to no error).
        if rng.random() < p_flip:
            flipped_qubit = rng.integers(1, 4)  # pick qubit 1, 2, or 3

            if current_error == 0:
                # No error active → this qubit is now flipped
                current_error = flipped_qubit
            elif current_error == flipped_qubit:
                # Same qubit flips again → error cancels out
                current_error = 0
            else:
                # Different qubit flips → now THAT one is the error
                # (simplified: we track only single-qubit errors)
                current_error = flipped_qubit

            flip_times.append(t)

        # ─── Step 2: Look up the true syndrome for current error ─
        s1_true, s2_true = ERROR_SIGNATURES[current_error]
        true_s1[t] = s1_true
        true_s2[t] = s2_true
        error_labels[t] = current_error

        # ─── Step 3: Generate noisy measurement record ──────────
        # r(t) = meas_strength * true_value + noise
        # meas_strength scales the signal, noise_std scales the noise
        # Their ratio is effectively your SNR
        r1[t] = meas_strength * s1_true + rng.normal(0, noise_std)
        r2[t] = meas_strength * s2_true + rng.normal(0, noise_std)

    return {
        "times":        times,
        "r1":           r1,
        "r2":           r2,
        "true_s1":      true_s1,
        "true_s2":      true_s2,
        "error_labels": error_labels,
        "flip_times":   flip_times,
    }


# ─── Batch Generator ──────────────────────────────────────────
def generate_dataset(
    n_trajectories: int = 1000,
    T: int = 200,
    dt: float = 0.01,
    p_flip: float = 0.01,
    meas_strength: float = 1.0,
    noise_std: float = 1.0,
    seed: int | None = None
) -> list[dict]:
    """
    Generate a batch of trajectories for training/testing.
    Each trajectory gets its own seed so results are reproducible.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_trajectories)

    dataset = []
    for i in range(n_trajectories):
        logical_state = rng.integers(0, 2)  # randomly pick |0_L> or |1_L>
        traj = generate_trajectory(
            T=T,
            dt=dt,
            p_flip=p_flip,
            meas_strength=meas_strength,
            noise_std=noise_std,
            logical_state=logical_state,
            seed=int(seeds[i])
        )
        dataset.append(traj)

    return dataset