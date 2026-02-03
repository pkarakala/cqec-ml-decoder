import numpy as np 

# Pauli matrices 

I = np.array([[1,0],[0,1]], dtype = complex)
X = np.array([[0,1],[1,0]], dtype = complex)
Z = np.array([[1,0],[0,-1]], dtype = complex)


# Tensor product helper 

def kron_list(ops: list[np.ndarray]) -> np.ndarray:
    #Tensor product of a list of 2x2 matrices 
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result 


# 3 - qubit stablizer code 

#S1 = Z1, Z2, I3 
S1 = kron_list([Z,Z,I])

#S2 = I1, Z2, Z3 
S2 = kron_list([I,Z,Z])

# Logical code works |000> and |111>

ket_0L = np.zeros(8, dtype = complex)
ket_0L[0] = 1.0

ket_1L = np.zeros(8, dtype = complex)
ket_1L[7] = 1.0

# Single - Qubit Bit Flip Errors in 3 qubit space 

# X on qubit 1 identity on 2, 3 
E1 = kron_list([X, I, I])

#X on qubit 2 
E2 = kron_list([I, X, I])

# X on qubit 3
E3 = kron_list([I, I, X])

# No error
E0 = kron_list([I, I, I])

ERRORS = [E0, E1, E2, E3]


""" 
Error Signature Table (look up table)

For each error, what (S1, S2) eigenvalue pair do you expect 
Eigen value = <psi| S | psi> after error is applied 

No error : (+1,+1)
Flip q1: (-1,+1)
Flip q2: (-1,-1)
Flip q3: (+1,-1)
"""
ERROR_SIGNATURES = {
    0: (+1, +1),
    1: (-1, +1),
    2: (-1, -1),
    3: (+1, -1),
}