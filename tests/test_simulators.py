import numpy as np

from src.sim_hamiltonian import generate_trajectory_hamiltonian
from src.sim_measurement import generate_trajectory


def test_measurement_simulator_output_structure_small_input():
    traj = generate_trajectory(T=8, seed=123)

    assert set(["times", "r1", "r2", "true_s1", "true_s2", "error_labels", "flip_times"]).issubset(traj)
    assert traj["r1"].shape == (8,)
    assert traj["r2"].shape == (8,)
    assert traj["error_labels"].shape == (8,)
    assert isinstance(traj["flip_times"], list)
    assert np.isfinite(traj["r1"]).all()
    assert np.isfinite(traj["r2"]).all()


def test_hamiltonian_simulator_includes_dynamic_fields_small_input():
    traj = generate_trajectory_hamiltonian(T=8, seed=123)

    assert traj["r1"].shape == (8,)
    assert traj["r2"].shape == (8,)
    assert traj["meas_strength_t"].shape == (8,)
    assert traj["drive_signal"].shape == (8,)
    assert np.isfinite(traj["meas_strength_t"]).all()
    assert np.isfinite(traj["drive_signal"]).all()
