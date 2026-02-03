import numpy as np
from .sim_measurement import generate_dataset



# ─── Windowing ────────────────────────────────────────────────
# The decoder doesn't see the entire trajectory at once.
# It sees a sliding window of the last W timesteps.
# This function slices a trajectory into overlapping windows
# so the decoder can learn from sequential context.
# ───────────────────────────────────────────────────────────────

def create_windows(trajectory: dict, window_size: int = 20) -> dict:
    """
    Takes a single trajectory and slices it into overlapping windows.

    For each timestep t >= window_size, we grab:
        - input:  [r1[t-W:t], r2[t-W:t]]  — the last W noisy readings
        - label:  error_labels[t-1]        — what error is active RIGHT NOW

    Returns:
        - X: array of shape (n_windows, window_size, 2)
             the 2 is [r1, r2] stacked as channels
        - y: array of shape (n_windows,)
             the error label at the end of each window
    """
    r1 = trajectory["r1"]
    r2 = trajectory["r2"]
    labels = trajectory["error_labels"]
    T = len(r1)

    X_list = []
    y_list = []

    for t in range(window_size, T):
        # Stack r1 and r2 into a (window_size, 2) chunk
        window = np.stack([
            r1[t - window_size:t],
            r2[t - window_size:t]
        ], axis=1)  # shape: (window_size, 2)

        X_list.append(window)
        y_list.append(labels[t - 1])  # label at the end of the window

    X = np.array(X_list)  # shape: (n_windows, window_size, 2)
    y = np.array(y_list)  # shape: (n_windows,)

    return {"X": X, "y": y}


# ─── Train / Test Split ───────────────────────────────────────
# We split at the TRAJECTORY level, not the window level.
# This is critical: if we split windows from the same trajectory
# into train and test, the model could memorize the noise pattern
# of that specific trajectory. Splitting by trajectory ensures
# the test set has truly unseen noise realizations.
# ───────────────────────────────────────────────────────────────

def build_train_test(
    n_trajectories: int = 1000,
    T: int = 200,
    window_size: int = 20,
    p_flip: float = 0.01,
    meas_strength: float = 1.0,
    noise_std: float = 1.0,
    test_fraction: float = 0.2,
    seed: int = 42
) -> dict:
    """
    End-to-end pipeline: simulate → window → split.

    Returns:
        - X_train, y_train: training inputs and labels
        - X_test,  y_test:  test inputs and labels
        - params:           the hyperparameters used (for logging)
    """
    # ── Step 1: generate all trajectories ───────────────────────
    dataset = generate_dataset(
        n_trajectories=n_trajectories,
        T=T,
        p_flip=p_flip,
        meas_strength=meas_strength,
        noise_std=noise_std,
        seed=seed
    )

    # ── Step 2: figure out the split index ──────────────────────
    n_test = int(n_trajectories * test_fraction)
    n_train = n_trajectories - n_test

    # ── Step 3: window each trajectory separately ──────────────
    train_X, train_y = [], []
    test_X,  test_y  = [], []

    for i, traj in enumerate(dataset):
        windowed = create_windows(traj, window_size=window_size)

        if i < n_train:
            train_X.append(windowed["X"])
            train_y.append(windowed["y"])
        else:
            test_X.append(windowed["X"])
            test_y.append(windowed["y"])

    # ── Step 4: concatenate all trajectories into single arrays ─
    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_test  = np.concatenate(test_X,  axis=0)
    y_test  = np.concatenate(test_y,  axis=0)

    # ── Step 5: log what we used ────────────────────────────────
    params = {
        "n_trajectories": n_trajectories,
        "n_train":        n_train,
        "n_test":         n_test,
        "T":              T,
        "window_size":    window_size,
        "p_flip":         p_flip,
        "meas_strength":  meas_strength,
        "noise_std":      noise_std,
        "test_fraction":  test_fraction,
        "seed":           seed,
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
        "params":  params,
    }