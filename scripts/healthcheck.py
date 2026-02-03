from src.sim_measurement import generate_dataset
from src.datasets import create_windows
from src.decoders import ThresholdDecoder, GRUDecoder
from src.metrics import evaluate_all

# Generate a tiny dataset
traj = generate_dataset(n_trajectories=1, T=60, seed=0)[0]
w = create_windows(traj, window_size=10)

# Baseline decoder
th = ThresholdDecoder()
th_preds = th.predict(w["X"])

# Dummy GRU (untrained) just to verify forward pass works
gru = GRUDecoder()

# Evaluate
out = evaluate_all(w["X"], w["y"], gru_model=gru)
print("Healthcheck OK. Keys:", out.keys())
print("Threshold acc:", out["threshold"]["accuracy"])
print("GRU acc:", out["gru"]["accuracy"])
