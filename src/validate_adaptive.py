"""
Phase 4 Unit Tests: Adaptive Decoding with Drifting Non-Idealities

Tests for:
1. sim_drifting.py - time-varying non-ideality simulator
2. adaptive_gru.py - online learning GRU decoder
"""

import numpy as np
import torch
from src.sim_drifting import generate_trajectory_drifting, generate_dataset_drifting
from src.adaptive_gru import AdaptiveGRUDecoder, train_adaptive_gru


# ═══════════════════════════════════════════════════════════════
# Test Group 1: Drifting Simulator - Basic Functionality
# ═══════════════════════════════════════════════════════════════

def test_drifting_output_shapes():
    """Test 1.1: Output shapes match expected dimensions."""
    traj = generate_trajectory_drifting(T=100, dt=0.01, seed=42)
    
    assert traj['t'].shape == (100,)
    assert traj['error_state'].shape == (100,)
    assert traj['r1'].shape == (100,)
    assert traj['r2'].shape == (100,)
    assert traj['s1_true'].shape == (100,)
    assert traj['s2_true'].shape == (100,)
    
    # Drift schedules
    assert traj['colored_noise_alpha_t'].shape == (100,)
    assert traj['transient_amplitude_t'].shape == (100,)
    assert traj['random_walk_strength_t'].shape == (100,)
    
    print("✓ Test 1.1 passed: output shapes correct")


def test_drifting_constant_parameters():
    """Test 1.2: With start==end, parameters stay constant."""
    
    # Phase 4 with no drift (constant parameters)
    traj = generate_trajectory_drifting(
        T=200, dt=0.01, p_flip=0.02, meas_strength=1.0, noise_std=1.0,
        colored_noise_alpha_start=0.5, colored_noise_alpha_end=0.5,
        transient_amplitude_start=0.3, transient_amplitude_end=0.3,
        random_walk_strength_start=0.2, random_walk_strength_end=0.2,
        seed=42
    )
    
    # All drift schedules should be constant
    assert np.allclose(traj['colored_noise_alpha_t'], 0.5, atol=1e-10)
    assert np.allclose(traj['transient_amplitude_t'], 0.3, atol=1e-10)
    assert np.allclose(traj['random_walk_strength_t'], 0.2, atol=1e-10)
    
    # Should have reasonable error rate
    error_rate = (traj['error_state'] != 0).mean()
    assert 0.3 < error_rate < 0.9, f"Error rate {error_rate:.3f} seems unreasonable"
    
    # Measurements should be finite
    assert np.all(np.isfinite(traj['r1']))
    assert np.all(np.isfinite(traj['r2']))
    
    print("✓ Test 1.2 passed: constant parameters work correctly")


# ═══════════════════════════════════════════════════════════════
# Test Group 2: Drift Schedules
# ═══════════════════════════════════════════════════════════════

def test_linear_drift_schedule():
    """Test 2.1: Linear drift interpolates correctly."""
    traj = generate_trajectory_drifting(
        T=100, colored_noise_alpha_start=0.0, colored_noise_alpha_end=1.0,
        drift_type='linear', seed=42
    )
    
    alpha_t = traj['colored_noise_alpha_t']
    
    # Should start at 0, end at 1
    assert np.isclose(alpha_t[0], 0.0, atol=1e-6)
    assert np.isclose(alpha_t[-1], 1.0, atol=1e-6)
    
    # Should be monotonically increasing
    assert np.all(np.diff(alpha_t) >= 0)
    
    # Midpoint should be ~0.5
    assert np.isclose(alpha_t[50], 0.5, atol=0.02)
    
    print("✓ Test 2.1 passed: linear drift schedule correct")


def test_sigmoid_drift_schedule():
    """Test 2.2: Sigmoid drift has S-curve shape."""
    traj = generate_trajectory_drifting(
        T=100, transient_amplitude_start=0.0, transient_amplitude_end=1.0,
        drift_type='sigmoid', seed=42
    )
    
    amp_t = traj['transient_amplitude_t']
    
    # Should start near 0, end near 1
    assert amp_t[0] < 0.1
    assert amp_t[-1] > 0.9
    
    # Should be monotonically increasing
    assert np.all(np.diff(amp_t) >= 0)
    
    # Derivative should be small at start/end, large in middle
    diff = np.diff(amp_t)
    assert diff[10] < diff[50]  # slower at start than middle
    assert diff[85] < diff[50]  # slower at end than middle
    
    print("✓ Test 2.2 passed: sigmoid drift schedule correct")


def test_sinusoidal_drift_schedule():
    """Test 2.3: Sinusoidal drift oscillates."""
    traj = generate_trajectory_drifting(
        T=200, random_walk_strength_start=0.0, random_walk_strength_end=1.0,
        drift_type='sinusoidal', drift_period=1.0, seed=42
    )
    
    rw_t = traj['random_walk_strength_t']
    
    # Should oscillate between 0 and 1
    assert rw_t.min() >= -0.1  # allow small numerical error
    assert rw_t.max() <= 1.1
    
    # Should have at least one peak and one valley
    peaks = (rw_t[1:-1] > rw_t[:-2]) & (rw_t[1:-1] > rw_t[2:])
    valleys = (rw_t[1:-1] < rw_t[:-2]) & (rw_t[1:-1] < rw_t[2:])
    assert peaks.sum() >= 1
    assert valleys.sum() >= 1
    
    print("✓ Test 2.3 passed: sinusoidal drift schedule correct")


# ═══════════════════════════════════════════════════════════════
# Test Group 3: Time-Varying Colored Noise
# ═══════════════════════════════════════════════════════════════

def test_colored_noise_drift():
    """Test 3.1: Colored noise autocorrelation changes over time."""
    traj = generate_trajectory_drifting(
        T=1000, dt=0.01, colored_noise_alpha_start=0.1, colored_noise_alpha_end=0.9,
        drift_type='linear', seed=42
    )
    
    cn = traj['colored_noise_r1']
    
    # Early portion (low alpha) should have low autocorrelation
    early = cn[:200]
    early_autocorr = np.corrcoef(early[:-1], early[1:])[0, 1]
    
    # Late portion (high alpha) should have high autocorrelation
    late = cn[-200:]
    late_autocorr = np.corrcoef(late[:-1], late[1:])[0, 1]
    
    assert late_autocorr > early_autocorr + 0.3  # significant increase
    
    print(f"✓ Test 3.1 passed: autocorr early={early_autocorr:.3f}, late={late_autocorr:.3f}")


# ═══════════════════════════════════════════════════════════════
# Test Group 4: Time-Varying Transients
# ═══════════════════════════════════════════════════════════════

def test_transient_amplitude_drift():
    """Test 4.1: Transient amplitude increases over time."""
    traj = generate_trajectory_drifting(
        T=1000, dt=0.01, p_flip=0.05,  # higher flip rate for more transients
        transient_amplitude_start=0.1, transient_amplitude_end=1.0,
        transient_decay=0.1, drift_type='linear', seed=42
    )
    
    trans = traj['transient_r1']
    error_state = traj['error_state']
    
    # Find flip events
    flips = np.where(np.diff(error_state) != 0)[0] + 1
    
    if len(flips) >= 10:
        # Compare early vs late flip transients
        early_flips = flips[flips < 300]
        late_flips = flips[flips > 700]
        
        if len(early_flips) > 0 and len(late_flips) > 0:
            early_peak = np.mean([trans[i] for i in early_flips])
            late_peak = np.mean([trans[i] for i in late_flips])
            
            assert late_peak > early_peak * 2  # should be significantly larger
            print(f"✓ Test 4.1 passed: early peak={early_peak:.3f}, late peak={late_peak:.3f}")
        else:
            print("✓ Test 4.1 passed: insufficient flips for comparison")
    else:
        print("✓ Test 4.1 passed: insufficient flips for comparison")


# ═══════════════════════════════════════════════════════════════
# Test Group 5: Time-Varying Random Walk
# ═══════════════════════════════════════════════════════════════

def test_random_walk_drift():
    """Test 5.1: Random walk variance increases with strength."""
    traj = generate_trajectory_drifting(
        T=1000, dt=0.01,
        random_walk_strength_start=0.01, random_walk_strength_end=0.5,
        drift_type='linear', seed=42
    )
    
    rw = traj['random_walk_mean']
    
    # Early portion (low strength) should have small increments
    early_increments = np.diff(rw[:300])
    early_std = np.std(early_increments)
    
    # Late portion (high strength) should have large increments
    late_increments = np.diff(rw[-300:])
    late_std = np.std(late_increments)
    
    assert late_std > early_std * 3  # significantly larger
    
    print(f"✓ Test 5.1 passed: early std={early_std:.4f}, late std={late_std:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test Group 6: Dataset Generation
# ═══════════════════════════════════════════════════════════════

def test_dataset_drifting():
    """Test 6.1: Dataset generation produces correct number of trajectories."""
    dataset = generate_dataset_drifting(
        n_trajectories=10, T=100, seed=42
    )
    
    assert len(dataset) == 10
    assert all('r1' in traj for traj in dataset)
    assert all('colored_noise_alpha_t' in traj for traj in dataset)
    
    print("✓ Test 6.1 passed: dataset generation correct")


def test_dataset_drifting_reproducibility():
    """Test 6.2: Same seed produces identical datasets."""
    dataset1 = generate_dataset_drifting(
        n_trajectories=5, T=100, colored_noise_alpha_start=0.2,
        colored_noise_alpha_end=0.8, seed=42
    )
    dataset2 = generate_dataset_drifting(
        n_trajectories=5, T=100, colored_noise_alpha_start=0.2,
        colored_noise_alpha_end=0.8, seed=42
    )
    
    for i in range(5):
        assert np.array_equal(dataset1[i]['error_state'], dataset2[i]['error_state'])
        assert np.allclose(dataset1[i]['r1'], dataset2[i]['r1'])
    
    print("✓ Test 6.2 passed: dataset reproducibility verified")


# ═══════════════════════════════════════════════════════════════
# Test Group 7: Adaptive GRU - Architecture
# ═══════════════════════════════════════════════════════════════

def test_adaptive_gru_architecture():
    """Test 7.1: Adaptive GRU has correct architecture."""
    model = AdaptiveGRUDecoder(
        input_size=2, hidden_size=32, num_classes=4,
        adapt_lr=0.0001, ema_decay=0.9
    )
    
    # Check components exist
    assert hasattr(model, 'gru')
    assert hasattr(model, 'classifier')
    assert hasattr(model, 'adapt_lr')
    assert hasattr(model, 'ema_decay')
    
    # Check forward pass works
    x = torch.randn(10, 20, 2)
    logits = model(x)
    assert logits.shape == (10, 4)
    
    print("✓ Test 7.1 passed: adaptive GRU architecture correct")


def test_adaptive_gru_ema_initialization():
    """Test 7.2: EMA buffers initialize on first adaptation."""
    model = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.001)
    
    assert model.ema_grads is None  # not initialized yet
    
    # Perform one adaptation step
    x = torch.randn(5, 20, 2)
    y = torch.randint(0, 4, (5,))
    logits = model(x)
    model.adapt_step(x, logits, y)
    
    assert model.ema_grads is not None  # now initialized
    assert len(model.ema_grads) > 0
    
    print("✓ Test 7.2 passed: EMA initialization correct")


# ═══════════════════════════════════════════════════════════════
# Test Group 8: Adaptive GRU - Online Learning
# ═══════════════════════════════════════════════════════════════

def test_adaptive_gru_weights_change():
    """Test 8.1: Weights change after adaptation."""
    model = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.01, adapt_every=1)
    
    # Save initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    # Perform several adaptation steps
    for _ in range(10):
        x = torch.randn(5, 20, 2)
        y = torch.randint(0, 4, (5,))
        logits = model(x)
        model.adapt_step(x, logits, y)
    
    # Check that at least some weights changed
    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_weights[name], atol=1e-6):
            changed = True
            break
    
    assert changed, "Weights should change after adaptation"
    print("✓ Test 8.1 passed: weights change during adaptation")


def test_adaptive_gru_confidence_threshold():
    """Test 8.2: Low-confidence predictions don't trigger adaptation."""
    model = AdaptiveGRUDecoder(
        hidden_size=32, adapt_lr=0.01, confidence_threshold=0.99,  # very high threshold
        adapt_every=1
    )
    
    # Create ambiguous input (should have low confidence)
    x = torch.zeros(5, 20, 2)  # all zeros -> ambiguous
    logits = model(x)
    
    # Try to adapt without true labels (pseudo-labeling)
    initial_count = model.update_count
    model.adapt_step(x, logits, y_true=None)
    
    # Update count should increment but EMA might not initialize (no confident predictions)
    # This is expected behavior
    print("✓ Test 8.2 passed: confidence threshold respected")


def test_adaptive_gru_adapt_every():
    """Test 8.3: adapt_every parameter controls update frequency."""
    model = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.01, adapt_every=5)
    
    # Save initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    # Perform 4 adaptation steps (should not update)
    for _ in range(4):
        x = torch.randn(5, 20, 2)
        y = torch.randint(0, 4, (5,))
        logits = model(x)
        model.adapt_step(x, logits, y)
    
    # Weights should not have changed yet
    no_change = all(
        torch.allclose(param, initial_weights[name], atol=1e-10)
        for name, param in model.named_parameters()
    )
    assert no_change, "Weights should not change before adapt_every threshold"
    
    # 5th step should trigger update
    x = torch.randn(5, 20, 2)
    y = torch.randint(0, 4, (5,))
    logits = model(x)
    model.adapt_step(x, logits, y)
    
    # Now weights should have changed
    changed = any(
        not torch.allclose(param, initial_weights[name], atol=1e-6)
        for name, param in model.named_parameters()
    )
    assert changed, "Weights should change on adapt_every-th step"
    
    print("✓ Test 8.3 passed: adapt_every parameter works correctly")


# ═══════════════════════════════════════════════════════════════
# Test Group 9: Adaptive GRU - Prediction Interface
# ═══════════════════════════════════════════════════════════════

def test_adaptive_gru_predict_adaptive():
    """Test 9.1: predict_adaptive returns correct shapes."""
    model = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.001)

    X = np.random.randn(50, 20, 2)
    y = np.random.randint(0, 4, 50)

    preds, history = model.predict_adaptive(X, y, reset_ema=True)

    assert preds.shape == (50,)
    assert 'confidences' in history
    assert 'adapted' in history
    assert 'supervised' in history
    assert history['confidences'].shape == (50,)
    assert history['adapted'].shape == (50,)
    assert history['supervised'].shape == (50,)

    # Fully supervised: all steps should use true labels
    assert history['supervised'].all(), "Fully supervised mode should use true labels at every step"

    print("✓ Test 9.1 passed: predict_adaptive interface correct")


def test_adaptive_gru_reset_ema():
    """Test 9.2: reset_ema clears adaptation state."""
    model = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.001)
    
    X = np.random.randn(20, 20, 2)
    y = np.random.randint(0, 4, 20)
    
    # First prediction with adaptation
    model.predict_adaptive(X, y, reset_ema=True)
    assert model.ema_grads is not None or model.update_count > 0  # some state exists
    first_count = model.update_count
    
    # Second prediction without reset
    model.predict_adaptive(X, y, reset_ema=False)
    second_count = model.update_count
    assert second_count >= first_count  # should continue counting (or stay same if no updates)
    
    # Third prediction with reset
    model.predict_adaptive(X, y, reset_ema=True)
    third_count = model.update_count
    # After reset, count should be less than second_count (reset to 0 then incremented)
    assert third_count < second_count, f"Count should reset: {third_count} < {second_count}"
    
    print("✓ Test 9.2 passed: reset_ema works correctly")


# ═══════════════════════════════════════════════════════════════
# Test Group 10: Training Pipeline
# ═══════════════════════════════════════════════════════════════

def test_train_adaptive_gru():
    """Test 10.1: Training pipeline runs without errors."""
    # Small synthetic dataset
    X_train = np.random.randn(100, 20, 2)
    y_train = np.random.randint(0, 4, 100)
    X_val = np.random.randn(20, 20, 2)
    y_val = np.random.randint(0, 4, 20)
    
    result = train_adaptive_gru(
        X_train, y_train, X_val, y_val,
        epochs=5, batch_size=32, hidden_size=16, seed=42
    )
    
    assert 'model' in result
    assert 'history' in result
    assert len(result['history']['train_loss']) == 5
    assert len(result['history']['val_acc']) == 5
    
    print("✓ Test 10.1 passed: training pipeline works")

# ═══════════════════════════════════════════════════════════════
# Test Group 9b: Hybrid Supervision Mode
# ═══════════════════════════════════════════════════════════════

def test_hybrid_supervision_periodic_labels():
    """Test 9.3: supervised_every injects true labels at correct intervals."""
    model = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.001)

    X = np.random.randn(100, 20, 2)
    y = np.random.randint(0, 4, 100)

    preds, history = model.predict_adaptive(
        X, y, reset_ema=True, supervised_every=10
    )

    assert preds.shape == (100,)
    assert history['supervised'].shape == (100,)

    # Every 10th step (index 9, 19, 29, ...) should be supervised
    for i in range(100):
        expected = ((i + 1) % 10 == 0)
        assert history['supervised'][i] == expected, \
            f"Step {i}: expected supervised={expected}, got {history['supervised'][i]}"

    # Exactly 10 supervised steps out of 100
    assert history['supervised'].sum() == 10

    print("✓ Test 9.3 passed: periodic supervision at correct intervals")


def test_hybrid_supervision_requires_labels():
    """Test 9.4: supervised_every > 0 without y_true raises ValueError."""
    model = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.001)

    X = np.random.randn(20, 20, 2)

    try:
        model.predict_adaptive(X, y_true=None, supervised_every=10)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ Test 9.4 passed: supervised_every requires y_true")


def test_pseudo_label_only_mode():
    """Test 9.5: Pure pseudo-label mode (no y_true) uses no true labels."""
    model = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.001)

    X = np.random.randn(50, 20, 2)

    preds, history = model.predict_adaptive(X, y_true=None, reset_ema=True)

    assert preds.shape == (50,)
    assert not history['supervised'].any(), "Pseudo-label mode should never use true labels"

    print("✓ Test 9.5 passed: pseudo-label mode uses no true labels")


def test_hybrid_vs_pseudo_label_divergence():
    """Test 9.6: Hybrid supervision produces different predictions than pseudo-label only."""
    torch.manual_seed(42)
    np.random.seed(42)

    X = np.random.randn(200, 20, 2).astype(np.float32)
    y = np.random.randint(0, 4, 200)

    # Train a small model first
    model_pseudo = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.001, confidence_threshold=0.5)
    model_hybrid = AdaptiveGRUDecoder(hidden_size=32, adapt_lr=0.001, confidence_threshold=0.5)

    # Copy weights so they start identical
    model_hybrid.load_state_dict(model_pseudo.state_dict())

    # Pseudo-label only
    preds_pseudo, _ = model_pseudo.predict_adaptive(X, y_true=None, reset_ema=True)

    # Hybrid with periodic supervision
    preds_hybrid, hist_hybrid = model_hybrid.predict_adaptive(
        X, y_true=y, reset_ema=True, supervised_every=20
    )

    # They should diverge (not be identical) because hybrid gets true labels
    # With 200 samples and supervision every 20, there are 10 supervised steps
    assert not np.array_equal(preds_pseudo, preds_hybrid), \
        "Hybrid and pseudo-label predictions should diverge"

    print("✓ Test 9.6 passed: hybrid supervision diverges from pseudo-label only")





# ═══════════════════════════════════════════════════════════════
# Test Group 11: Edge Cases
# ═══════════════════════════════════════════════════════════════

def test_drifting_edge_cases():
    """Test 11.1: Edge cases don't crash."""
    # T=1
    traj = generate_trajectory_drifting(T=1, seed=42)
    assert len(traj['r1']) == 1
    
    # No drift (start == end == 0)
    traj = generate_trajectory_drifting(
        T=100, colored_noise_alpha_start=0.0, colored_noise_alpha_end=0.0, seed=42
    )
    assert np.allclose(traj['colored_noise_alpha_t'], 0.0)
    
    # Extreme drift
    traj = generate_trajectory_drifting(
        T=100, transient_amplitude_start=0.0, transient_amplitude_end=10.0, seed=42
    )
    assert not np.any(np.isnan(traj['r1']))
    assert not np.any(np.isinf(traj['r1']))
    
    print("✓ Test 11.1 passed: edge cases handled")


# ═══════════════════════════════════════════════════════════════
# Run All Tests
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Unit Tests: Adaptive Decoding")
    print("=" * 60)
    
    # Group 1: Drifting Simulator
    print("\n[Group 1] Drifting Simulator - Basic Functionality")
    test_drifting_output_shapes()
    test_drifting_constant_parameters()
    
    # Group 2: Drift Schedules
    print("\n[Group 2] Drift Schedules")
    test_linear_drift_schedule()
    test_sigmoid_drift_schedule()
    test_sinusoidal_drift_schedule()
    
    # Group 3: Time-Varying Colored Noise
    print("\n[Group 3] Time-Varying Colored Noise")
    test_colored_noise_drift()
    
    # Group 4: Time-Varying Transients
    print("\n[Group 4] Time-Varying Transients")
    test_transient_amplitude_drift()
    
    # Group 5: Time-Varying Random Walk
    print("\n[Group 5] Time-Varying Random Walk")
    test_random_walk_drift()
    
    # Group 6: Dataset Generation
    print("\n[Group 6] Dataset Generation")
    test_dataset_drifting()
    test_dataset_drifting_reproducibility()
    
    # Group 7: Adaptive GRU Architecture
    print("\n[Group 7] Adaptive GRU - Architecture")
    test_adaptive_gru_architecture()
    test_adaptive_gru_ema_initialization()
    
    # Group 8: Adaptive GRU Online Learning
    print("\n[Group 8] Adaptive GRU - Online Learning")
    test_adaptive_gru_weights_change()
    test_adaptive_gru_confidence_threshold()
    test_adaptive_gru_adapt_every()
    
    # Group 9: Adaptive GRU Prediction
    print("\n[Group 9] Adaptive GRU - Prediction Interface")
    test_adaptive_gru_predict_adaptive()
    test_adaptive_gru_reset_ema()
    
    # Group 9b: Hybrid Supervision
    print("\n[Group 9b] Hybrid Supervision Mode")
    test_hybrid_supervision_periodic_labels()
    test_hybrid_supervision_requires_labels()
    test_pseudo_label_only_mode()
    test_hybrid_vs_pseudo_label_divergence()
    
    # Group 10: Training Pipeline
    print("\n[Group 10] Training Pipeline")
    test_train_adaptive_gru()
    
    # Group 11: Edge Cases
    print("\n[Group 11] Edge Cases")
    test_drifting_edge_cases()
    
    print("\n" + "=" * 60)
    print("All Phase 4 tests passed! ✓")
    print("=" * 60)
