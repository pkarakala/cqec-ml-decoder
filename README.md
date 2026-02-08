# Continuous Quantum Error Correction with Machine Learning Decoders

**Authors:** Pranav Reddy (preddy@ucsb.edu) Clark Enge (clarkenge@ucsb.edu)

A research project exploring machine learning approaches to continuous quantum error correction, comparing data-driven decoders against principled baselines on a 3-qubit repetition code with realistic time-dependent dynamics.

---

## Project Overview

This project implements a complete pipeline for simulating continuous syndrome measurements in a quantum error-correcting code and benchmarking multiple decoding strategies. Rather than discrete syndrome extraction, we simulate noisy analog readout signals and train neural networks to infer error states directly from time-series data.

**Phase 1** establishes the phenomenological baseline with static syndrome signals and random bit-flip errors. We compare a simple threshold decoder against a GRU-based recurrent neural network.

**Phase 2** (current) extends to physically grounded continuous dynamics with time-dependent Hamiltonians, including coherent drive oscillations, calibration drift, and measurement backaction. We introduce a Bayesian filter as a principled baseline and demonstrate that the GRU adapts better to model mismatch.

---

## Repository Structure
```
cqec_phase1/
├── src/                    # Core modules
│   ├── operators.py        # Pauli matrices, stabilizers, error signatures
│   ├── sim_measurement.py  # Phase 1: static syndrome simulator
│   ├── sim_hamiltonian.py  # Phase 2: time-dependent Hamiltonian simulator
│   ├── datasets.py         # Windowing and train/test splits
│   ├── decoders.py         # Threshold and GRU decoders
│   ├── bayesian_filter.py  # Wonham filter / HMM decoder
│   ├── metrics.py          # Accuracy, confusion matrices, latency
│   └── test_*.py           # Unit tests
├── notebooks/              # Jupyter notebooks
│   ├── 01_phase1_setup.ipynb
│   └── 02_phase2_dynamics.ipynb
├── outputs/                # Generated figures and results
│   └── figures/
└── requirements.txt
```

---

## Module Descriptions

### Core Simulation
- **`operators.py`** — Defines quantum building blocks: Pauli matrices, stabilizers S₁ and S₂, logical codewords |0⟩ₗ and |1⟩ₗ, single-qubit bit-flip errors, and the error signature lookup table.

- **`sim_measurement.py`** — Phase 1 physics simulator. Generates continuous noisy analog readout signals r₁(t) and r₂(t) with random bit-flip errors.

- **`sim_hamiltonian.py`** — Phase 2 physics simulator. Adds time-dependent Hamiltonian dynamics: rotating drive (coherent oscillations), calibration drift (time-varying measurement strength), and measurement backaction (quantum noise).

### Data Processing
- **`datasets.py`** — Converts raw trajectories into windowed train/test splits. Handles both Phase 1 and Phase 2 data formats.

### Decoders
- **`decoders.py`** — Threshold decoder (average-based heuristic) and GRU decoder (recurrent neural network).

- **`bayesian_filter.py`** — Wonham filter / hidden Markov model. Optimal decoder under Phase 1 assumptions (static syndromes, Gaussian noise, Markov transitions).

### Evaluation
- **`metrics.py`** — Computes accuracy, per-class accuracy, confusion matrices, and detection latency.

### Testing
- **`test_operators.py`** — 44 unit tests verifying quantum operator math.
- **`test_hamiltonian.py`** — 58 unit tests for Phase 2 simulator.
- **`test_bayesian.py`** — 22 unit tests for Bayesian filter.

---

## Installation
```bash
# Clone the repository
git clone https://github.com/pkarakala/cqec-ml-decoder.git
cd cqec-ml-decoder

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running Unit Tests
```bash
# Test quantum operators
python3 -m src.test_operators

# Test Phase 2 simulator
python3 -m src.test_hamiltonian

# Test Bayesian filter
python3 -m src.test_bayesian
```

### Running Notebooks
```bash
jupyter notebook notebooks/01_phase1_setup.ipynb
jupyter notebook notebooks/02_phase2_dynamics.ipynb
```

---

## Key Results

### Phase 1 (Static Syndromes)
- **Threshold decoder:** ~86% accuracy
- **GRU decoder:** ~96% accuracy
- GRU learns temporal structure in continuous measurements, outperforming static thresholding especially at high noise levels.

### Phase 2 (Time-Dependent Dynamics)
- **Threshold decoder:** ~85% accuracy
- **Bayesian filter:** ~94% accuracy (optimal under model assumptions)
- **GRU decoder:** ~96% accuracy (adapts to model mismatch)
- When drive/drift violate Bayesian model assumptions, the GRU maintains higher robustness by learning dynamics directly from data.

---

## Future Work (Phase 3)

- Graph neural network decoder exploiting stabilizer code topology
- Correlated noise models (cross-talk between qubits)
- Scaling to larger codes (5-qubit, 7-qubit surface code patches)
- Latency-optimized real-time decoding

---

## Dependencies

- Python 3.10+
- NumPy
- PyTorch
- Matplotlib
- SciPy
- Jupyter

See `requirements.txt` for exact versions.

---

## License

This project is for research and educational purposes.
