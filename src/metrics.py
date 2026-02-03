import numpy as np
import torch
from .decoders import ThresholdDecoder, GRUDecoder



# ─── Core Metrics ─────────────────────────────────────────────
# These functions take predictions and ground truth labels
# and compute the numbers that go in your results section.
# All of them work on numpy arrays so they're decoder-agnostic —
# you can pass in predictions from the threshold decoder or the
# GRU or anything else.
# ───────────────────────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of timesteps where the predicted error matches
    the true error exactly.
    """
    return (y_true == y_pred).mean()


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Accuracy broken down by error class.
    Tells you if the decoder is good at all classes or just
    biased toward one (e.g. always predicting 'no error').

    Returns dict like: {0: 0.98, 1: 0.91, 2: 0.88, 3: 0.92}
    """
    results = {}
    for cls in range(4):
        mask = y_true == cls                # find all timesteps where true label is cls
        if mask.sum() == 0:
            results[cls] = None             # class never appeared — skip it
        else:
            results[cls] = (y_pred[mask] == cls).mean()  # accuracy on just that class
    return results


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    4x4 matrix where entry [i, j] = number of times
    true label was i but predicted label was j.
    Rows = true, Columns = predicted.
    Diagonal = correct predictions.
    """
    mat = np.zeros((4, 4), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        mat[true_label, pred_label] += 1
    return mat


# ─── Detection Latency ────────────────────────────────────────
# When an error actually happens, how many timesteps does it
# take for the decoder to notice? This is one of the most
# important metrics for real-time QEC — you need to catch
# errors fast or they propagate.
# ───────────────────────────────────────────────────────────────

def compute_latency(
    error_labels: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 20
) -> dict:
    """
    For each flip event in the trajectory, measures how many
    timesteps pass between the flip and the first correct
    detection by the decoder.

    error_labels: the true error label at every timestep (from trajectory)
    y_pred:       the decoder's predictions aligned to the same timesteps
    window_size:  offset to account for the windowing delay

    Returns:
        - latencies:      list of latency values (one per detected flip)
        - mean_latency:   average latency across all flips
        - median_latency: median latency
        - n_flips:        total number of flip events found
    """
    latencies = []

    # Find flip events: timesteps where error_label changes
    for t in range(1, len(error_labels)):
        if error_labels[t] != error_labels[t - 1]:
            # A flip just happened at timestep t
            new_error = error_labels[t]

            # Now scan forward from t to find when the decoder first
            # predicts the new error correctly
            # We start scanning at t - window_size + 1 because that's
            # the earliest window that could contain timestep t
            scan_start = max(0, t - window_size + 1)

            detected = False
            for t2 in range(scan_start, min(t + window_size, len(y_pred))):
                if y_pred[t2] == new_error:
                    latency = t2 - t + window_size  # account for window offset
                    latencies.append(latency)
                    detected = True
                    break

            if not detected:
                # Decoder never caught this error within the search window
                latencies.append(None)

    # Filter out None values for stats
    valid_latencies = [l for l in latencies if l is not None]

    return {
        "latencies":      latencies,
        "mean_latency":   np.mean(valid_latencies) if valid_latencies else None,
        "median_latency": np.median(valid_latencies) if valid_latencies else None,
        "n_flips":        len(latencies),
        "n_detected":     len(valid_latencies),
    }


# ─── Full Evaluation Pipeline ─────────────────────────────────
# Runs both decoders on the test set, computes all metrics,
# and returns everything in one clean dict. This is the one
# function you'll call from your notebook to get results.
# ───────────────────────────────────────────────────────────────

def evaluate_all(
    X_test: np.ndarray,
    y_test: np.ndarray,
    gru_model: GRUDecoder,
    trajectories: list[dict] | None = None
) -> dict:
    """
    Runs ThresholdDecoder and GRUDecoder on the test set,
    computes accuracy, per-class accuracy, confusion matrices,
    and latency if trajectories are provided.

    trajectories: optional list of raw trajectory dicts from the simulator.
                  needed for latency computation. if None, latency is skipped.

    Returns a nested dict:
        {
            "threshold": { "accuracy", "per_class", "confusion", "latency" },
            "gru":       { "accuracy", "per_class", "confusion", "latency" },
        }
    """
    results = {}

    # ── Threshold decoder predictions ───────────────────────────
    threshold = ThresholdDecoder()
    threshold_preds = threshold.predict(X_test)

    results["threshold"] = {
        "accuracy":    accuracy(y_test, threshold_preds),
        "per_class":   per_class_accuracy(y_test, threshold_preds),
        "confusion":   confusion_matrix(y_test, threshold_preds),
    }

    # ── GRU decoder predictions ─────────────────────────────────
    gru_model.eval()
    with torch.no_grad():
        X_tensor  = torch.tensor(X_test, dtype=torch.float32)
        logits    = gru_model(X_tensor)
        gru_preds = logits.argmax(dim=1).numpy()

    results["gru"] = {
        "accuracy":    accuracy(y_test, gru_preds),
        "per_class":   per_class_accuracy(y_test, gru_preds),
        "confusion":   confusion_matrix(y_test, gru_preds),
    }

    # ── Latency (only if raw trajectories provided) ─────────────
    if trajectories is not None:
        # Run latency on each test trajectory individually
        threshold_latencies = []
        gru_latencies       = []

        for traj in trajectories:
            T = len(traj["error_labels"])
            window_size = X_test.shape[1]

            # Re-predict on this single trajectory's windows
            from .datasets import create_windows
            w = create_windows(traj, window_size=window_size)

            th_preds  = threshold.predict(w["X"])
            with torch.no_grad():
                gru_logits = gru_model(torch.tensor(w["X"], dtype=torch.float32))
                gr_preds   = gru_logits.argmax(dim=1).numpy()

            th_lat = compute_latency(traj["error_labels"], th_preds, window_size)
            gr_lat = compute_latency(traj["error_labels"], gr_preds, window_size)

            threshold_latencies.append(th_lat)
            gru_latencies.append(gr_lat)

        results["threshold"]["latency"] = threshold_latencies
        results["gru"]["latency"]       = gru_latencies

    return results