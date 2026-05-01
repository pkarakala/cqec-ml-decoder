import numpy as np

from src.datasets import create_windows


def test_create_windows_uses_past_measurements_and_end_label():
    trajectory = {
        "r1": np.array([0, 1, 2, 3, 4]),
        "r2": np.array([10, 11, 12, 13, 14]),
        "error_labels": np.array([0, 1, 2, 3, 0]),
    }

    windowed = create_windows(trajectory, window_size=3)

    assert windowed["X"].shape == (2, 3, 2)
    assert windowed["y"].shape == (2,)
    np.testing.assert_array_equal(
        windowed["X"][0],
        np.array([[0, 10], [1, 11], [2, 12]]),
    )
    np.testing.assert_array_equal(windowed["y"], np.array([2, 3]))
