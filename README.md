operators.py — Defines the quantum building blocks: Pauli matrices, the two stabilizer operators S1 and S2, the logical codewords |0_L⟩ and |1_L⟩, the three single-qubit bit-flip errors, and the error signature lookup table that maps each error to its unique (S1, S2) measurement pair. Everything else in the project imports from here.

sim_measurement.py — The physics simulator. Takes the operators and runs a continuous measurement simulation over time. At each timestep it randomly injects bit-flip errors and generates noisy analog readout signals r₁(t) and r₂(t) that mimic what a real quantum detector would produce. Also has a batch generator for producing large training datasets.

datasets.py — Takes the raw trajectories from the simulator and packages them into clean train/test splits with the correct input/output format that the decoders expect. Handles windowing (slicing the time series into fixed-length chunks) and label alignment.

decoders.py — Contains the actual decoding algorithms that try to recover which error is active from the noisy measurement records. Phase 1 has a simple threshold baseline and an RNN (GRU). Phase 2 will add the Bayesian filter and GNN.

metrics.py — Evaluates how well the decoders are doing. Computes detection accuracy, latency (how fast an error is detected after it occurs), and other metrics. Produces the numbers that go in your results section.

test_operators.py — Unit tests for operators.py. 44 tests that verify all the matrix math is correct before anything else runs.