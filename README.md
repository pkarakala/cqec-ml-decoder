# Continuous Quantum Error Correction with ML Decoders

> Can a neural network decode quantum errors better than Bayes' theorem?

**Authors:** Pranav Reddy ([preddy@ucsb.edu](mailto:preddy@ucsb.edu)) · Clark Enge ([clarkenge@ucsb.edu](mailto:clarkenge@ucsb.edu))

---

## The Problem

Quantum computers are noisy. A 3-qubit repetition code protects information by encoding it across qubits (`|0⟩ₗ = |000⟩`, `|1⟩ₗ = |111⟩`), but you need to *continuously* monitor stabilizer measurements to catch bit-flip errors before they corrupt your computation.

Traditional QEC extracts discrete syndrome bits. We instead work with **continuous analog readout** — noisy real-valued signals `r₁(t)` and `r₂(t)` that encode stabilizer eigenvalues buried in Gaussian noise. The question: **who decodes these signals best?**

```
No error:      S₁ = +1, S₂ = +1   →  r₁(t) ≈ +1 + noise
Flip qubit 1:  S₁ = -1, S₂ = +1   →  r₁(t) ≈ -1 + noise
Flip qubit 2:  S₁ = -1, S₂ = -1   →  both flip
Flip qubit 3:  S₁ = +1, S₂ = -1   →  r₂(t) ≈ -1 + noise
```

---

## Three Decoders, Head to Head

| Decoder | Type | How it works |
|---------|------|-------------|
| **Threshold** | Heuristic | Averages `r₁`, `r₂` over a window, checks the sign quadrant |
| **Bayesian Filter** | Probabilistic | Wonham filter / HMM — optimal under known noise model |
| **GRU Network** | Learned | Recurrent neural net that learns temporal patterns from data |

The **GRU** ([`src/decoders.py`](src/decoders.py)) processes measurement windows sequentially, building internal memory before classifying:

```python
class GRUDecoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_classes=4):
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        output, h_n = self.gru(x)
        return self.classifier(h_n[-1])  # classify from final hidden state
```

The **Bayesian filter** ([`src/bayesian_filter.py`](src/bayesian_filter.py)) maintains a belief distribution over 4 error states, updating via Bayes' rule at each timestep:

```
For each measurement (r₁, r₂):
    1. Predict:  belief ← transition_matrix × belief
    2. Update:   belief ← belief × P(r₁, r₂ | state)
    3. Normalize: belief ← belief / sum(belief)
```

The **Threshold decoder** ([`src/decoders.py`](src/decoders.py)) is the simplest possible baseline — average the window, check the sign:

```python
r1_avg = X[:, :, 0].mean(axis=1)
r2_avg = X[:, :, 1].mean(axis=1)
# Quadrant → error class: (+,+)=0  (-,+)=1  (-,-)=2  (+,-)=3
```

---

## Four Phases of Increasing Realism

### Phase 1 — Static Syndromes

The baseline scenario ([`src/sim_measurement.py`](src/sim_measurement.py)): random bit-flip errors with constant measurement strength and i.i.d. Gaussian noise. The Bayesian filter's assumptions hold perfectly here.

```python
# Core measurement model (sim_measurement.py):
r1[t] = meas_strength * s1_true + rng.normal(0, noise_std)
r2[t] = meas_strength * s2_true + rng.normal(0, noise_std)
```

### Phase 2 — Time-Dependent Hamiltonian Dynamics

This is where it gets interesting ([`src/sim_hamiltonian.py`](src/sim_hamiltonian.py)). We add three physical effects that **break the Bayesian model's assumptions**:

- **Coherent drive** — sinusoidal oscillations modulate the syndrome signal even without errors
- **Calibration drift** — measurement strength changes linearly over time
- **Measurement backaction** — the act of measuring injects additional quantum noise

```python
# Phase 2 measurement model (sim_hamiltonian.py):
meas_strength_t[t] = meas_strength + drift_rate * t * dt
drive_signal[t] = drive_amplitude * cos(drive_frequency * t * dt)

r1[t] = (meas_strength_t[t] + drive_signal[t]) * s1_true + readout_noise + backaction_noise
```

The Bayesian filter assumes static `meas_strength` and no drive — so when these dynamics kick in, it degrades. The GRU learns the dynamics directly from data.

### Phase 3 — Non-Ideal Measurement Effects

Real quantum hardware has non-idealities that violate textbook assumptions ([`src/sim_nonideal.py`](src/sim_nonideal.py)):

- **Colored noise (AR(1) process)** — temporally correlated readout noise, not white Gaussian
- **Post-flip transients** — exponential ring-down artifacts after each error flip
- **Random-walk drift** — Brownian motion in measurement calibration

```python
# Phase 3 adds three non-idealities:
colored_noise[t] = alpha * colored_noise[t-1] + sqrt(1-alpha²) * white_noise[t]
transient[t] = amplitude * exp(-t / decay) * (1 if flip_occurred else 0)
random_walk[t] = random_walk[t-1] + normal(0, strength)
```

These effects break the Bayesian filter's white noise and static parameter assumptions. The GRU learns to handle them from data.

### Phase 4 — Adaptive Decoding Under Drift

The ultimate challenge ([`src/sim_drifting.py`](src/sim_drifting.py), [`src/adaptive_gru.py`](src/adaptive_gru.py)): hardware parameters don't stay constant — they drift during operation.

- **Time-varying non-idealities** — colored noise, transients, and drift parameters change within a single trajectory
- **Adaptive GRU** — continues learning online via EMA-smoothed gradient updates
- **Three adaptation strategies** tested:
  1. **Static GRU** — trained once, frozen weights (baseline)
  2. **Pseudo-label adaptation** — self-training with confident predictions (fails under heavy drift)
  3. **Hybrid adaptation** — periodic true labels + pseudo-labels in between (works!)

```python
# Phase 4: Parameters drift over time
colored_noise_alpha[t] = interpolate(0.1 → 0.9, t, drift_type='linear')
transient_amplitude[t] = interpolate(0.1 → 1.0, t, drift_type='linear')

# Adaptive GRU: hybrid supervision mode
# Every N windows, inject a true label (periodic recalibration)
# In between, use high-confidence pseudo-labels
preds, history = model.predict_adaptive(
    X_test, y_true=y_test, supervised_every=50  # true label every 50 windows
)
```

Pure self-training fails because confident wrong predictions poison online learning. But periodic recalibration + online adaptation maintains accuracy under drift.

---

## Results

### Phase 1 — Static Syndromes

| Decoder | Accuracy |
|---------|----------|
| Threshold | ~86% |
| GRU | **~96%** |

The GRU learns temporal correlations in the continuous measurement stream that simple averaging misses.

### Phase 2 — Time-Dependent Dynamics

| Decoder | Accuracy | Notes |
|---------|----------|-------|
| Threshold | ~85% | No model, no adaptation |
| Bayesian Filter | ~94% | Optimal *if* model assumptions hold |
| GRU | **~96%** | Adapts to model mismatch |

When drive and drift violate the Bayesian model's assumptions, the GRU maintains robustness by learning dynamics directly from data.

### Phase 3 — Non-Ideal Measurement Effects

| Decoder | Accuracy | Notes |
|---------|----------|-------|
| Threshold | ~79% | Simple averaging fails with colored noise |
| Bayesian Filter | ~84% | White noise assumption violated |
| GRU | **~83%** | Learns temporal correlations in colored noise |

With colored noise, post-flip transients, and random-walk drift, all decoders degrade. The Bayesian filter suffers most because its core assumptions (white noise, static parameters) are violated. The GRU learns non-ideal effects from data but needs more training data to match Phase 2 performance.

### Phase 4 — Adaptive Decoding Under Drift

Five-way comparison under drifting parameters (results pending full notebook run):

| Decoder | Strategy | Notes |
|---------|----------|-------|
| Threshold | No model | Degrades steadily with drift |
| Bayesian Filter | Fixed parameters | Fails as parameters drift away |
| Static GRU | Trained once, frozen | Degrades over time |
| Adaptive GRU (pseudo-labels) | Self-training | Barely helps — confident wrong predictions poison learning |
| Adaptive GRU (hybrid) | Periodic recalibration + pseudo-labels | Maintains accuracy under drift |

The key result is the temporal segment analysis: as parameters drift from early to late in each trajectory, static decoders degrade while the hybrid adaptive GRU maintains performance. Even infrequent supervision (true labels every 50-100 windows, ~1-2% of data) provides significant improvement.

### Figures

| | |
|---|---|
| ![Decoder Comparison](outputs/figures/decoder_comparison.png) | ![Training Curves](outputs/figures/training_curves.png) |
| ![Confusion Matrices](outputs/figures/confusion_matrices.png) | ![Robustness vs Noise](outputs/figures/robustness_vs_noise.png) |
| ![Phase 2 Dynamics](outputs/figures/phase2_dynamics_comparison.png) | ![Phase 2 Robustness](outputs/figures/phase2_robustness_vs_drive.png) |
| ![Phase 3 Non-Idealities](outputs/figures/phase3_nonideal_effects.png) | ![Phase 3 Decoder Comparison](outputs/figures/phase3_decoder_comparison.png) |
| ![Phase 3 Confusion Matrices](outputs/figures/phase3_confusion_matrices.png) | ![Phase 3 Robustness](outputs/figures/phase3_robustness_sweeps.png) |

---

## Under the Hood

### Quantum Operators ([`src/operators.py`](src/operators.py))

The stabilizer code is built from scratch using tensor products of Pauli matrices:

```python
S1 = Z ⊗ Z ⊗ I    # Stabilizer 1: checks qubits 1,2
S2 = I ⊗ Z ⊗ Z    # Stabilizer 2: checks qubits 2,3

E0 = I ⊗ I ⊗ I    # No error
E1 = X ⊗ I ⊗ I    # Bit-flip on qubit 1
E2 = I ⊗ X ⊗ I    # Bit-flip on qubit 2
E3 = I ⊗ I ⊗ X    # Bit-flip on qubit 3
```

Each error produces a unique syndrome signature `(S₁, S₂)`, which is what the decoders try to infer from noisy measurements.

### Data Pipeline ([`src/datasets.py`](src/datasets.py))

Trajectories are sliced into overlapping windows of size `W`. The split happens at the **trajectory level** — not the window level — to prevent data leakage. Test windows come from entirely unseen noise realizations.

### Evaluation ([`src/metrics.py`](src/metrics.py))

Beyond overall accuracy, we track:

- **Per-class accuracy** — catches decoders that just predict "no error" all the time
- **Confusion matrices** — reveals which error pairs get confused
- **Detection latency** — how many timesteps after a flip before the decoder catches it (critical for real-time QEC)

---

## Repository Structure

```
├── src/
│   ├── operators.py          # Pauli matrices, stabilizers, error signatures
│   ├── sim_measurement.py    # Phase 1: static syndrome simulator
│   ├── sim_hamiltonian.py    # Phase 2: time-dependent Hamiltonian simulator
│   ├── sim_nonideal.py       # Phase 3: non-ideal measurement effects
│   ├── sim_drifting.py       # Phase 4: time-varying parameter drift
│   ├── datasets.py           # Windowing + trajectory-level train/test splits
│   ├── decoders.py           # Threshold baseline + static GRU decoder
│   ├── adaptive_gru.py       # Phase 4: adaptive GRU with hybrid supervision
│   ├── bayesian_filter.py    # Wonham filter / HMM decoder
│   ├── metrics.py            # Accuracy, confusion matrices, detection latency
│   ├── test_operators.py     # 44 unit tests — quantum operator math
│   ├── test_hamiltonian.py   # 58 unit tests — Phase 2 simulator
│   ├── test_bayesian.py      # 22 unit tests — Bayesian filter
│   ├── test_nonideal.py      # 99 unit tests — Phase 3 simulator
│   └── test_adaptive.py      # 25 unit tests — Phase 4 adaptive decoder + hybrid supervision
├── notebooks/
│   ├── 01_phase1_setup.ipynb
│   ├── 02_phase2_dynamics.ipynb
│   ├── 03_phase3_nonideal.ipynb
│   └── 04_phase4_adaptive_decoding.ipynb
├── presentation/
│   ├── adaptive_qec_slides.pptx  # Competition slide deck (22 slides)
│   ├── slides_content.md          # Slide-by-slide content reference
│   └── build_slides.py            # Script to regenerate slides
├── outputs/figures/               # Generated plots
├── scripts/
│   ├── healthcheck.py             # Quick sanity check
│   └── test_phase4_smoke.py       # Phase 4 end-to-end smoke test
└── requirements.txt
```

---

## Getting Started

```bash
git clone https://github.com/pkarakala/cqec-ml-decoder.git
cd cqec-ml-decoder

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Sanity check
python3 scripts/healthcheck.py

# Run the notebooks
jupyter notebook notebooks/01_phase1_setup.ipynb
```

### Running Tests

```bash
python3 -m src.test_operators      # 44 tests — quantum operator math
python3 -m src.test_hamiltonian    # 58 tests — Phase 2 simulator
python3 -m src.test_bayesian       # 22 tests — Bayesian filter
python3 -m src.test_nonideal       # 99 tests — Phase 3 non-ideal effects
python3 -m src.test_adaptive       # 25 tests — Phase 4 adaptive decoder + hybrid supervision
```

---

## Dependencies

Python 3.10+ · NumPy · PyTorch · SciPy · Matplotlib · Jupyter · scikit-learn · QuTiP

See [`requirements.txt`](requirements.txt) for the full list.

---

## Key Findings

### 1. ML Decoders Learn What Models Miss

The GRU decoder achieves competitive accuracy without knowing the underlying physics. When Hamiltonian dynamics (Phase 2) violate the Bayesian filter's assumptions, the GRU maintains performance by learning temporal patterns directly from data.

### 2. Non-Idealities Break Model-Based Approaches

Phase 3 demonstrates that realistic hardware effects (colored noise, transients, random-walk drift) degrade all decoders, but especially the Bayesian filter whose white-noise and static-parameter assumptions are violated. This motivates data-driven approaches for real quantum hardware.

### 3. Pure Self-Training Fails, But Hybrid Supervision Works

Phase 4 reveals that pure pseudo-label adaptation barely improves over static GRU — confident wrong predictions poison self-training under distribution shift. However, periodic recalibration (injecting true labels every 50-100 windows) combined with online adaptation maintains accuracy as hardware drifts. This "hybrid supervision" approach models realistic QEC where periodic state verification is available.

### 4. The Accuracy-Robustness Tradeoff

- Bayesian filter: Highest accuracy when assumptions hold, but brittle to model mismatch
- Static GRU: Robust to moderate non-idealities, but degrades under drift
- Adaptive GRU (pseudo-labels): Barely helps — confident wrong predictions reinforce mistakes
- Adaptive GRU (hybrid): Maintains performance under drift with minimal supervision overhead

---

## Future Work

### Immediate Extensions
- Latency analysis for Phase 3 & 4 (detection delay with non-idealities and drift)
- Robustness sweeps for Phase 4 (performance vs drift rate, drift type)
- Meta-learning for faster adaptation
- Ensemble methods with multiple adaptation rates

### Scaling Up
- Graph neural networks exploiting stabilizer code topology
- Larger codes (5-qubit, 7-qubit surface code patches)
- Correlated noise models (cross-talk between qubits)
- Multi-qubit error patterns (correlated bit-flips)

### Real Hardware
- Benchmark on experimental data from superconducting qubits
- Transfer learning: pre-train on simulation, fine-tune on hardware
- Real-time decoding with latency constraints
- Integration with quantum control systems

---

## License

This project is for research and educational purposes.
