import numpy as np
import sympy as sp
import math
import random

def d_w(w=64):
    """
    Find the smallest N such that the 2-adic valuation of N! is greater than or equal to w.
    """
    curr = 1
    N = 1
    while not curr >= w:
        N += 1
        curr = v_2(N)
    return N

def v_2(n):
    """
    Computes the 2-adic valuation of n! using Legendre's formula.
    """
    return sum([math.floor(n / 2**i) for i in range(1, math.floor(math.log(n, 2)) + 1)])

def G_j(j, N=64):
    """
    Computes the zero polynomial generator for degree j in the ring 2^N.
    """
    basis = falling_factorial_basis(j, N)
    c_j = 2 ** max(N-v_2(j), 0)
    ret = c_j * basis[:,j]
    return np.flip(ret)

def domain(N=64):
    """
    Generates the domain of the ring 2^N.
    """
    match N:
        case 8:
            return 'uint8'
        case 16:
            return 'uint16'
        case 32:
            return 'uint32'
        case 64:
            return 'uint64'
        case _:
            raise ValueError("Only 2^8, 2^16, 2^32, and 2^64 are supported.")

def falling_factorial_basis(degree, mod_exp=64):
    """
    Generate a falling factorial basis of degree `d` for the permutation polynomials mod 2^N.

    This implementation directly relies on the underlying storage of the polynomial coefficients in NumPy
    to compute mod 2^N.
    """

    dtype = domain(mod_exp)
    mtx = np.zeros(shape=(degree+1, degree+1), dtype=dtype)
    mtx[0][0] = 1 # x^(0) = 1
    prev = [1]
    for i in range(1, degree + 1):
        # Multiply the previous polynomial by (x - i) to get the basis polynomial of degree i
        newp = np.polymul(prev, [1, -(i+1)]) # x^(i) = (x - (i-1))*x^(i-1)

        # Store polynomial as column vector with leading coefficient at the bottom
        # so resultant matrix is triangular and is guaranteed to have inverse mod 2^N
        # (since the diagonal is all 1s the determinant is 1 and hence nonzero)
        mtx[0:i+1,i] = np.flip(newp)
        prev = newp
        
    return mtx

def modular_inv_tri(M):
    """
    Compute the modular inverse of a triaprint(np.polynomial.polynomial(poly))

        1 a b         1 a' b'                                                           1 0 0
    A = 0 1 c  A^-1 = 0 1  c'  when multiplied together should give the identity matrix 0 1 0
        0 0 1         0 0  1                                                            0 0 1


    The dot product 
        
        A[0,:]  * A^-1[:, 1] = 0
        [1 a b] * [a' 1 0] = 0
        1*a' + a = 0
        a' = -a

    Hence, each element will be equal to the additive inverse of the dot product of the remaining elements in the vector.

    """
    N = M.shape[0]
    inv = np.eye(N, dtype=M.dtype)
    row_base = 0
    col_base = 1
    for i in range(N-1, 0, -1):
        for j in range(0, i):
            row = row_base + j
            col = col_base + j
            vecM = M[row,:]
            vecMinv = inv[:, col]
            inv[row][col] = -(np.dot(vecM, vecMinv))
        col_base += 1
            
    return inv

def generate_zero_polynomial(max_degree, N=64):
    """
    Generates a zero polynomial in the ring 2^N of degree <= max_degree by
    choosing random coefficients to multiply by the zero ideal basis of degree `max_degree`.

    H(x) = Sum_{j=0}^{d} h_j x^(j) wheren x^(j) is the falling factorial at j.

    h_j ~ 0 mod (2 ^ max(w-v_2(j!), 0)) by taking the jth discrete derivative of H(x) and evaluating at 0.

    Note v_2(j!) is the 2-adic valuation of j!.

    """
    poly = np.array([0], dtype=domain(N))
    for i in range(2, d_w(max_degree), 2):
        component_i = random.randint(0, 2**N-1) * G_j(i, N)
        poly = np.polyadd(poly, component_i)
    return poly

def generate_univariate_permutation_polynomial(max_degree, N=64):
    """
    Generates a univariate permutation polynomial in the ring 2^N of degree <= max_degree.
    """
    pass

def generate_multivariate_permutation_polynomial(max_degree, num_variables, N=64):
    """
    Generates a multivariate permutation polynomial in the ring 2^N of degree <= max_degree
    with `num_variables` variables.
    """
    pass


if __name__ == "__main__":
    np.seterr(over='ignore')
    a = falling_factorial_basis(8, 9)
    print(a)
    a_inv = modular_inv_tri(a)
    print(a_inv)
    print(np.matmul(a, a_inv))