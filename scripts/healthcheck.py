import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.sim_measurement import generate_dataset
from src.datasets import create_windows
from src.decoders import ThresholdDecoder, GRUDecoder
from src.metrics import accuracy

# Generate a tiny dataset
traj = generate_dataset(n_trajectories=1, T=60, seed=0)[0]
w = create_windows(traj, window_size=10)

# Baseline decoder
th = ThresholdDecoder()
th_preds = th.predict(w["X"])

# Dummy GRU (untrained) just to verify forward pass works
gru = GRUDecoder()
with torch.no_grad():
    logits = gru(torch.tensor(w["X"], dtype=torch.float32))

print("Healthcheck OK.")
print("Threshold acc:", accuracy(w["y"], th_preds))
print("GRU forward logits shape:", tuple(logits.shape))
