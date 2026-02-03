import numpy as np
import sys
from operators import (I, X, Z, kron_list, S1, S2,
                       ket_0L, ket_1L, E0, E1, E2, E3,
                       ERRORS, ERROR_SIGNATURES)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}")
        failed += 1

print("\n=== TEST 1: Matrix Shapes ===")
check("I is 2x2",   I.shape == (2, 2))
check("X is 2x2",   X.shape == (2, 2))
check("Z is 2x2",   Z.shape == (2, 2))
check("S1 is 8x8",  S1.shape == (8, 8))
check("S2 is 8x8",  S2.shape == (8, 8))
check("E0 is 8x8",  E0.shape == (8, 8))
check("E1 is 8x8",  E1.shape == (8, 8))
check("ket_0L is length 8", ket_0L.shape == (8,))
check("ket_1L is length 8", ket_1L.shape == (8,))

print("\n=== TEST 2: Pauli Properties ===")
check("X² = I",     np.allclose(X @ X, I))
check("Z² = I",     np.allclose(Z @ Z, I))
check("XZ = -ZX",   np.allclose(X @ Z, -Z @ X))

print("\n=== TEST 3: Stabilizers are Hermitian & Square to Identity ===")
check("S1 is Hermitian",  np.allclose(S1, S1.conj().T))
check("S2 is Hermitian",  np.allclose(S2, S2.conj().T))
check("S1² = I (8x8)",    np.allclose(S1 @ S1, np.eye(8)))
check("S2² = I (8x8)",    np.allclose(S2 @ S2, np.eye(8)))

print("\n=== TEST 4: Codewords are Valid (Stabilizer Eigenvalue +1) ===")
check("S1 |0_L⟩ = +|0_L⟩",  np.allclose(S1 @ ket_0L, ket_0L))
check("S2 |0_L⟩ = +|0_L⟩",  np.allclose(S2 @ ket_0L, ket_0L))
check("S1 |1_L⟩ = +|1_L⟩",  np.allclose(S1 @ ket_1L, ket_1L))
check("S2 |1_L⟩ = +|1_L⟩",  np.allclose(S2 @ ket_1L, ket_1L))

print("\n=== TEST 5: Error Signatures Match the Lookup Table ===")
for error_idx in range(4):
    E = ERRORS[error_idx]
    expected_s1, expected_s2 = ERROR_SIGNATURES[error_idx]

    errored_state = E @ ket_0L
    measured_s1 = np.real(errored_state.conj() @ S1 @ errored_state)
    measured_s2 = np.real(errored_state.conj() @ S2 @ errored_state)
    check(f"Error {error_idx} on |0_L⟩: S1={int(measured_s1):+d} (expected {expected_s1:+d})",
          np.isclose(measured_s1, expected_s1))
    check(f"Error {error_idx} on |0_L⟩: S2={int(measured_s2):+d} (expected {expected_s2:+d})",
          np.isclose(measured_s2, expected_s2))

    errored_state = E @ ket_1L
    measured_s1 = np.real(errored_state.conj() @ S1 @ errored_state)
    measured_s2 = np.real(errored_state.conj() @ S2 @ errored_state)
    check(f"Error {error_idx} on |1_L⟩: S1={int(measured_s1):+d} (expected {expected_s1:+d})",
          np.isclose(measured_s1, expected_s1))
    check(f"Error {error_idx} on |1_L⟩: S2={int(measured_s2):+d} (expected {expected_s2:+d})",
          np.isclose(measured_s2, expected_s2))

print("\n=== TEST 6: Errors are Distinct & Orthogonal Outputs ===")
for i in range(4):
    for j in range(i+1, 4):
        state_i = ERRORS[i] @ ket_0L
        state_j = ERRORS[j] @ ket_0L
        overlap = np.abs(state_i.conj() @ state_j)
        check(f"|⟨E{i}|0_L⟩ · E{j}|0_L⟩| = 0  (got {overlap:.4f})", np.isclose(overlap, 0.0))

print("\n=== TEST 7: kron_list Consistency ===")
manual_S1 = np.kron(np.kron(Z, Z), I)
manual_S2 = np.kron(np.kron(I, Z), Z)
check("kron_list S1 == manual S1", np.allclose(S1, manual_S1))
check("kron_list S2 == manual S2", np.allclose(S2, manual_S2))

print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}\n")

if failed > 0:
    sys.exit(1)
else:
    print("  All operators verified. Safe to build on.\n")