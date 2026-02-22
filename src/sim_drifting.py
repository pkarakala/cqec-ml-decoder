"""
Phase 4: Time-Drifting Non-Idealities Simulator

Extends Phase 3 by making non-ideality parameters drift WITHIN a single trajectory.
This models realistic hardware where calibration, noise characteristics, and
measurement response change over time due to temperature, aging, or other slow processes.

Key differences from Phase 3:
- Phase 3: non-idealities are constant within each trajectory (but vary between trajectories)
- Phase 4: non-idealities change smoothly over time within a single trajectory

This creates a challenging scenario where:
- Static GRU (trained on fixed parameters) will degrade as hardware drifts
- Bayesian filter (with fixed parameters) will also degrade
- Adaptive GRU (online learning) should maintain performance by tracking the drift
"""

import numpy as np
from src.sim_nonideal import generate_trajectory_nonideal


def generate_trajectory_drifting(
    T: int = 1000,
    dt: float = 0.01,
    p_flip: float = 0.02,
    meas_strength: float = 1.0,
    noise_std: float = 1.0,
    # Phase 2 dynamics (can still be used)
    drive_amplitude: float = 0.0,
    drive_frequency: float = 0.0,
    drift_rate: float = 0.0,
    backaction_strength: float = 0.0,
    # Phase 3 non-idealities - now with DRIFT
    colored_noise_alpha_start: float = 0.0,
    colored_noise_alpha_end: float = 0.0,
    transient_amplitude_start: float = 0.0,
    transient_amplitude_end: float = 0.0,
    transient_decay: float = 0.1,  # decay rate doesn't drift
    random_walk_strength_start: float = 0.0,
    random_walk_strength_end: float = 0.0,
    # Drift dynamics
    drift_type: str = 'linear',  # 'linear', 'sigmoid', 'sinusoidal'
    drift_period: float = 1.0,   # for sinusoidal drift (in units of T*dt)
    seed: int | None = None
) -> dict:
    """
    Generate a single trajectory where non-ideality parameters drift over time.
    
    Parameters
    ----------
    T : int
        Number of timesteps
    dt : float
        Time resolution
    p_flip : float
        Bit-flip probability per timestep
    meas_strength : float
        Base measurement strength
    noise_std : float
        Base white noise standard deviation
    
    Phase 2 dynamics (unchanged from Phase 3):
    drive_amplitude, drive_frequency, drift_rate, backaction_strength
    
    Phase 3 non-idealities (now with drift):
    colored_noise_alpha_start/end : float
        AR(1) coefficient drifts from start to end value
    transient_amplitude_start/end : float
        Post-flip transient amplitude drifts
    transient_decay : float
        Exponential decay rate (constant, doesn't drift)
    random_walk_strength_start/end : float
        Brownian drift strength parameter drifts
    
    Drift dynamics:
    drift_type : str
        'linear' - smooth linear interpolation from start to end
        'sigmoid' - smooth S-curve transition (slow-fast-slow)
        'sinusoidal' - periodic oscillation between start and end
    drift_period : float
        For sinusoidal drift, period in units of total time (T*dt)
    
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    dict with keys:
        't' : array (T,) - time points
        'error_state' : array (T,) - true error state at each timestep
        'r1', 'r2' : arrays (T,) - noisy measurement readouts
        's1_true', 's2_true' : arrays (T,) - true syndrome values
        
        # Drift schedules (new in Phase 4)
        'colored_noise_alpha_t' : array (T,) - time-varying alpha
        'transient_amplitude_t' : array (T,) - time-varying transient amplitude
        'random_walk_strength_t' : array (T,) - time-varying random walk strength
        
        # All Phase 3 decomposition fields still present
        'colored_noise_r1', 'colored_noise_r2' : arrays (T,)
        'transient_r1', 'transient_r2' : arrays (T,)
        'random_walk_mean' : array (T,)
        ... (all other Phase 3 fields)
    """
    rng = np.random.default_rng(seed)
    
    # ── Step 1: Generate drift schedules ──────────────────────────────────
    t_normalized = np.linspace(0, 1, T)  # normalized time [0, 1]
    
    if drift_type == 'linear':
        # Simple linear interpolation
        progress = t_normalized
    elif drift_type == 'sigmoid':
        # Smooth S-curve: slow at start/end, fast in middle
        # Using logistic function: 1 / (1 + exp(-k*(x - 0.5)))
        k = 10  # steepness parameter
        progress = 1 / (1 + np.exp(-k * (t_normalized - 0.5)))
        # Normalize to [0, 1]
        progress = (progress - progress[0]) / (progress[-1] - progress[0])
    elif drift_type == 'sinusoidal':
        # Periodic oscillation
        # Maps [0,1] to [start, end, start, end, ...] over drift_period cycles
        progress = 0.5 * (1 + np.sin(2 * np.pi * t_normalized / drift_period))
    else:
        raise ValueError(f"Unknown drift_type: {drift_type}")
    
    # Interpolate each parameter
    colored_noise_alpha_t = (colored_noise_alpha_start + 
                             progress * (colored_noise_alpha_end - colored_noise_alpha_start))
    transient_amplitude_t = (transient_amplitude_start + 
                             progress * (transient_amplitude_end - transient_amplitude_start))
    random_walk_strength_t = (random_walk_strength_start + 
                              progress * (random_walk_strength_end - random_walk_strength_start))
    
    # ── Step 2: Generate trajectory with time-varying parameters ──────────
    # We'll do this manually by calling Phase 3 simulator in chunks
    # and stitching together, OR we can implement the drift directly.
    # For simplicity and full control, let's implement directly.
    
    # Initialize arrays
    t = np.arange(T) * dt
    error_state = np.zeros(T, dtype=int)
    s1_true = np.ones(T)
    s2_true = np.ones(T)
    
    # Error signatures (same as Phase 1-3)
    ERROR_SIGNATURES = {
        0: (+1, +1),  # no error
        1: (-1, +1),  # flip qubit 1
        2: (-1, -1),  # flip qubit 2
        3: (+1, -1),  # flip qubit 3
    }
    
    # ── Step 3: Generate error process ────────────────────────────────────
    # Use same error generation as Phase 3 for compatibility
    current_error = 0
    for i in range(T):
        # Maybe inject a bit-flip (same logic as Phase 3)
        if rng.random() < p_flip:
            flipped_qubit = rng.integers(1, 4)
            if current_error == 0:
                current_error = flipped_qubit
            elif current_error == flipped_qubit:
                current_error = 0
            else:
                current_error = flipped_qubit
        
        error_state[i] = current_error
        s1_true[i], s2_true[i] = ERROR_SIGNATURES[current_error]
    
    # ── Step 4: Generate measurement signals with time-varying non-idealities ──
    
    # Base measurement (Phase 2 dynamics)
    meas_strength_t = meas_strength + drift_rate * t
    drive_signal = drive_amplitude * np.cos(2 * np.pi * drive_frequency * t)
    
    # White noise component
    white_noise_r1 = rng.normal(0, noise_std, T)
    white_noise_r2 = rng.normal(0, noise_std, T)
    
    # Backaction noise (Phase 2)
    backaction_noise_r1 = rng.normal(0, backaction_strength, T)
    backaction_noise_r2 = rng.normal(0, backaction_strength, T)
    
    # ── Colored noise with time-varying alpha ──
    colored_noise_r1 = np.zeros(T)
    colored_noise_r2 = np.zeros(T)
    colored_noise_r1[0] = white_noise_r1[0]
    colored_noise_r2[0] = white_noise_r2[0]
    
    for i in range(1, T):
        # AR(1) with time-varying coefficient
        alpha_t = colored_noise_alpha_t[i]
        colored_noise_r1[i] = alpha_t * colored_noise_r1[i-1] + np.sqrt(1 - alpha_t**2) * white_noise_r1[i]
        colored_noise_r2[i] = alpha_t * colored_noise_r2[i-1] + np.sqrt(1 - alpha_t**2) * white_noise_r2[i]
    
    # ── Post-flip transients with time-varying amplitude ──
    transient_r1 = np.zeros(T)
    transient_r2 = np.zeros(T)
    
    for i in range(1, T):
        if error_state[i] != error_state[i-1]:
            # Error flip detected - inject transient
            transient_r1[i] = transient_amplitude_t[i]
            transient_r2[i] = transient_amplitude_t[i]
        else:
            # Exponential decay
            transient_r1[i] = transient_r1[i-1] * np.exp(-transient_decay * dt)
            transient_r2[i] = transient_r2[i-1] * np.exp(-transient_decay * dt)
    
    # ── Random walk drift with time-varying strength ──
    random_walk_mean = np.zeros(T)
    for i in range(1, T):
        # Brownian increment with time-varying diffusion
        increment = rng.normal(0, random_walk_strength_t[i] * np.sqrt(dt))
        random_walk_mean[i] = random_walk_mean[i-1] + increment
    
    # ── Combine all components ──
    r1 = ((meas_strength_t + drive_signal + random_walk_mean) * s1_true + 
          colored_noise_r1 + transient_r1 + backaction_noise_r1)
    r2 = ((meas_strength_t + drive_signal + random_walk_mean) * s2_true + 
          colored_noise_r2 + transient_r2 + backaction_noise_r2)
    
    return {
        't': t,
        'error_state': error_state,
        'r1': r1,
        'r2': r2,
        's1_true': s1_true,
        's2_true': s2_true,
        # Drift schedules (new in Phase 4)
        'colored_noise_alpha_t': colored_noise_alpha_t,
        'transient_amplitude_t': transient_amplitude_t,
        'random_walk_strength_t': random_walk_strength_t,
        # Signal decomposition (Phase 3 compatibility)
        'meas_strength_t': meas_strength_t,
        'drive_signal': drive_signal,
        'colored_noise_r1': colored_noise_r1,
        'colored_noise_r2': colored_noise_r2,
        'transient_r1': transient_r1,
        'transient_r2': transient_r2,
        'random_walk_mean': random_walk_mean,
        'white_noise_r1': white_noise_r1,
        'white_noise_r2': white_noise_r2,
        'backaction_noise_r1': backaction_noise_r1,
        'backaction_noise_r2': backaction_noise_r2,
    }


def generate_dataset_drifting(
    n_trajectories: int = 100,
    T: int = 1000,
    dt: float = 0.01,
    p_flip: float = 0.02,
    meas_strength: float = 1.0,
    noise_std: float = 1.0,
    # Phase 2 dynamics
    drive_amplitude: float = 0.0,
    drive_frequency: float = 0.0,
    drift_rate: float = 0.0,
    backaction_strength: float = 0.0,
    # Phase 4 drifting non-idealities
    colored_noise_alpha_start: float = 0.0,
    colored_noise_alpha_end: float = 0.0,
    transient_amplitude_start: float = 0.0,
    transient_amplitude_end: float = 0.0,
    transient_decay: float = 0.1,
    random_walk_strength_start: float = 0.0,
    random_walk_strength_end: float = 0.0,
    drift_type: str = 'linear',
    drift_period: float = 1.0,
    seed: int | None = None
) -> list[dict]:
    """
    Generate a dataset of trajectories with drifting non-idealities.
    
    Each trajectory has the same drift schedule but different noise realizations.
    
    Returns
    -------
    list of dicts, each containing one trajectory (see generate_trajectory_drifting)
    """
    rng = np.random.default_rng(seed)
    trajectories = []
    
    for i in range(n_trajectories):
        traj_seed = None if seed is None else rng.integers(0, 2**31)
        traj = generate_trajectory_drifting(
            T=T, dt=dt, p_flip=p_flip,
            meas_strength=meas_strength, noise_std=noise_std,
            drive_amplitude=drive_amplitude, drive_frequency=drive_frequency,
            drift_rate=drift_rate, backaction_strength=backaction_strength,
            colored_noise_alpha_start=colored_noise_alpha_start,
            colored_noise_alpha_end=colored_noise_alpha_end,
            transient_amplitude_start=transient_amplitude_start,
            transient_amplitude_end=transient_amplitude_end,
            transient_decay=transient_decay,
            random_walk_strength_start=random_walk_strength_start,
            random_walk_strength_end=random_walk_strength_end,
            drift_type=drift_type,
            drift_period=drift_period,
            seed=traj_seed
        )
        trajectories.append(traj)
    
    return trajectories
