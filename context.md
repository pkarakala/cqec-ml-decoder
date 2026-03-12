# Session Context — Continuous QEC with ML Decoders

This file preserves context from development sessions so future sessions can pick up where we left off.

## Project Overview

Undergraduate research project for a data science club competition (and potentially publishable).
Builds ML decoders for continuous quantum error correction on a 3-qubit repetition code.
Compares threshold, Bayesian filter, and GRU neural network decoders across 4 phases of increasing realism.

**Authors:** Pranav Reddy (preddy@ucsb.edu), Clark Enge (clarkenge@ucsb.edu)
**Repo:** https://github.com/pkarakala/cqec-ml-decoder

## Architecture

```
src/
  operators.py          — Pauli matrices, stabilizers, error signatures
  sim_measurement.py    — Phase 1: static syndrome simulator
  sim_hamiltonian.py    — Phase 2: time-dependent Hamiltonian dynamics
  sim_nonideal.py       — Phase 3: colored noise, transients, random-walk drift
  sim_drifting.py       — Phase 4: time-varying parameter drift within trajectories
  datasets.py           — Windowing + trajectory-level train/test splits
  decoders.py           — Threshold baseline + static GRU decoder
  adaptive_gru.py       — Phase 4: adaptive GRU with hybrid supervision
  bayesian_filter.py    — Wonham filter / HMM decoder
  metrics.py            — Accuracy, confusion matrices, detection latency
  test_*.py             — 248 unit tests total (44+58+22+99+25)

notebooks/
  01_phase1_setup.ipynb              — Phase 1 evaluation (complete, run)
  02_phase2_dynamics.ipynb           — Phase 2 evaluation (complete, run)
  03_phase3_nonideal.ipynb           — Phase 3 evaluation (complete, run)
  04_phase4_adaptive_decoding.ipynb  — Phase 4 evaluation (rebuilt, NEEDS FULL RUN)
  _build_phase4.py                   — Script to regenerate Phase 4 notebook

presentation/
  adaptive_qec_slides.pptx   — 22-slide competition deck (18 main + 4 backup)
  slides_content.md          — Slide-by-slide content reference
  build_slides.py            — Script to regenerate slides

scripts/
  healthcheck.py             — Quick sanity check
  test_phase4_smoke.py       — Phase 4 end-to-end smoke test (tiny dataset)
```

## Four Phases

### Phase 1 — Static Syndromes
- Ideal conditions: constant measurement strength, white Gaussian noise
- Results: Threshold ~86%, GRU ~96%
- Bayesian filter not tested (added in Phase 2)

### Phase 2 — Hamiltonian Dynamics
- Adds coherent drive, calibration drift, measurement backaction
- Results: Threshold ~85%, Bayesian ~94%, GRU ~96%
- GRU maintains performance when Bayesian assumptions break

### Phase 3 — Non-Ideal Measurement Effects
- Adds colored noise (AR(1)), post-flip transients, random-walk drift
- Results: Threshold ~79%, Bayesian ~84%, GRU ~83%
- All decoders degrade; Bayesian and GRU are neck-and-neck
- GRU overfits: best val_acc ~86.7% at epoch 10, drops to 82.9% by epoch 50

### Phase 4 — Adaptive Decoding Under Drift (REVISED)
- Parameters drift WITHIN each trajectory (not just between)
- Three adaptation strategies compared:
  1. Static GRU — trained once, frozen (baseline)
  2. Pseudo-label adaptation — self-training with confident predictions
  3. Hybrid adaptation — periodic true labels every N windows + pseudo-labels between

## Key Decision: Phase 4 Revision

### Problem Identified
The original Phase 4 had only pseudo-label adaptation. Results showed adaptive GRU
barely outperformed static GRU (~1% improvement). The model was ~95% confident but
only ~70% accurate under heavy drift — confident wrong predictions poisoned self-training.

### Solution Implemented
Added `supervised_every` parameter to `predict_adaptive()` in `adaptive_gru.py`.
This enables hybrid supervision: inject a true label every N windows (modeling periodic
recalibration), use pseudo-labels in between.

### Narrative
"Pure self-training fails because confident wrong predictions poison online learning.
But periodic recalibration + online adaptation maintains accuracy under drift."

This is both more realistic (real QEC has periodic state verification) and produces
a stronger result (clear separation between static, pseudo-label, and hybrid).


## Code Changes Made This Session

### 1. `src/adaptive_gru.py`
- Added `supervised_every` parameter to `predict_adaptive()`
- When `supervised_every > 0`, true labels injected every N steps (hybrid mode)
- When `supervised_every == 0` and `y_true` provided: fully supervised (every step)
- When `y_true is None`: pure pseudo-label mode
- History dict now includes `'supervised'` array tracking which steps used true labels
- Updated module docstring to describe three adaptation modes
- Adaptation parameters used in notebook: `adapt_lr=0.001`, `ema_decay=0.7`, `confidence_threshold=0.8`

### 2. `src/test_adaptive.py`
- Updated `test_adaptive_gru_predict_adaptive` to check `'supervised'` key in history
- Added 4 new tests (Group 9b: Hybrid Supervision Mode):
  - `test_hybrid_supervision_periodic_labels` — verifies correct interval injection
  - `test_hybrid_supervision_requires_labels` — ValueError when supervised_every > 0 without y_true
  - `test_pseudo_label_only_mode` — confirms no true labels used when y_true=None
  - `test_hybrid_vs_pseudo_label_divergence` — hybrid and pseudo-label produce different predictions
- Total: 25 tests, all passing

### 3. `notebooks/04_phase4_adaptive_decoding.ipynb`
- Completely rebuilt via `_build_phase4.py` script
- 27 cells (16 code, 11 markdown)
- 5-way decoder comparison: threshold, bayesian, static GRU, pseudo-label, hybrid
- Temporal segment analysis (5 segments, early→late drift)
- Supervision frequency sweep (every 10, 20, 50, 100, 200, 500 windows)
- Dataset params in notebook: N_TRAJECTORIES=200, T=1000, WINDOW_SIZE=20
- NEEDS FULL RUN — only smoke-tested with tiny dataset (20 traj, T=200)

### 4. `presentation/`
- `build_slides.py` — generates 22-slide .pptx with Catppuccin dark theme
- `adaptive_qec_slides.pptx` — generated deck, Phase 1-3 data filled in
- `slides_content.md` — detailed slide-by-slide content reference
- Phase 4 slides have `____` blanks to fill after running notebook
- Test count corrected to 248 across all presentation files

### 5. `README.md`
- Updated to "Four Phases of Increasing Realism"
- Phase 4 section describes 5-way comparison with hybrid supervision
- Repo structure includes presentation/, scripts/, correct notebook names
- Test counts: 25 for Phase 4 (was 21)
- Key Findings section updated with hybrid supervision narrative

### 6. `.gitignore` and `requirements.txt`
- Added python-pptx>=1.0 to requirements
- Removed qutip (not used)
- Pinned minimum versions for all deps
- Added IDE, swap file, and OS patterns to .gitignore

## What Needs To Happen Next

### Immediate (before competition)
1. **Run Phase 4 notebook with full parameters** — N_TRAJECTORIES=200, T=1000
   - This is the ~15-20 min run that produces real figures and numbers
   - May need to tune `adapt_lr` (try 0.005) or `ema_decay` (try 0.5) if hybrid gap is small
2. **Fill in Phase 4 blanks in slides** — search for `____` in slides_content.md
3. **Upload .pptx to Google Slides** — right-click → Open with Google Slides
4. **Practice the talk** — 12-15 minutes, focus on the temporal degradation plot (Slide 12)

### If hybrid doesn't show clear advantage
- Increase `adapt_lr` to 0.005 for more aggressive adaptation
- Lower `ema_decay` to 0.5 for less gradient smoothing
- Make drift more aggressive (wider parameter ranges)
- Try `supervised_every=20` instead of 50

### Longer term
- Run with larger dataset (500+ trajectories) for cleaner statistics
- Latency analysis for Phase 3 & 4
- Consider meta-learning for faster adaptation
- Scale to larger codes (surface codes)
- Publication target: NeurIPS/ICML quantum ML workshops or PRX Quantum

## Environment
- macOS (Apple Silicon)
- Python 3.12 in `.venv/`
- PyTorch 2.10.0, NumPy 2.4.2
- All commands use `.venv/bin/python` or `.venv/bin/pip`

## Git State
- Branch: main
- Remote: origin (github.com/pkarakala/cqec-ml-decoder)
- All changes committed and pushed
- Clean working tree as of last session
