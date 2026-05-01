import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.decoders import GRUDecoder, ThresholdDecoder


def test_threshold_decoder_prediction_shape_and_type():
    X = np.array([
        [[1.0, 1.0], [0.8, 0.9]],
        [[-1.0, 1.0], [-0.7, 0.8]],
        [[-1.0, -1.0], [-0.8, -0.9]],
        [[1.0, -1.0], [0.7, -0.8]],
    ])

    preds = ThresholdDecoder().predict(X)

    assert preds.shape == (4,)
    assert preds.dtype.kind in {"i", "u"}
    np.testing.assert_array_equal(preds, np.array([0, 1, 2, 3]))


def test_gru_decoder_forward_shape_without_training():
    model = GRUDecoder(hidden_size=8, dropout=0.0)
    X = torch.zeros((3, 5, 2), dtype=torch.float32)

    with torch.no_grad():
        logits = model(X)

    assert tuple(logits.shape) == (3, 4)
