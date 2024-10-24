import numpy as np
import sympy as sp
import math
import random

class BinaryPolynomial:
    
    def __init__(self, coeffs, mod_exp=64):
        """
        Initialize a polynomial with coefficients `coeffs` in the ring 2^N.

        Coefficients are stored with leading coefficient at the top of the array.
        """
        self.mod_exp = mod_exp

        match self.mod_exp:
            case 8:
                self.dtype = 'uint8'
            case 16:
                self.dtype = 'uint16'
            case 32:
                self.dtype = 'uint32'
            case 64:
                self.dtype = 'uint64'
            case _:
                raise ValueError("Only 2^8, 2^16, 2^32, and 2^64 are supported.")
            
        self.coeffs = np.array(coeffs, dtype=self.dtype)
    
    def degree(self):
        return len(self.coeffs) - 1
    
    def derivative(self):
        return BinaryPolynomial(np.polyder(self.coeffs), mod_exp=self.mod_exp)
    
    def compose(self, other):
        return BinaryPolynomial(np.polyval(np.poly1d(self.coeffs), np.poly1d(other.coeffs)), mod_exp=self.mod_exp)
    
    def truncate(self):
        self.coeffs = np.trim_zeros(self.coeffs, 'f')

    def is_identity(self):
        return np.array_equal(np.array([1, 0], dtype=self.dtype), self.coeffs)

    def newton_inverse(self, zero_ideal):
        """
        Compute the Newton inverse of a polynomial in the ring 2^N.
        """

        # Initial guess is f(^-1)(x) = x
        polyx = BinaryPolynomial([1, 0], mod_exp=self.mod_exp)

        g_i = polyx
        iter_count = 0

        while True: 
            assert(iter_count < self.degree()*3)

            composition = self.compose(g_i)

            composition = zero_ideal.simplify(composition)
            
            if composition.is_identity():
                break

            composition -= polyx


            dg = g_i.derivative()

            g_i -= dg * composition
            
            g_i = zero_ideal.simplify(g_i)
            iter_count += 1
            
        return g_i

    def __add__(self, other):
        return BinaryPolynomial(np.polyadd(self.coeffs, other.coeffs), mod_exp=self.mod_exp)
    
    def __sub__(self, other):
        return BinaryPolynomial(np.polysub(self.coeffs, other.coeffs), mod_exp=self.mod_exp)
    
    def __mul__(self, other):
        return BinaryPolynomial(np.polymul(self.coeffs, other.coeffs), mod_exp=self.mod_exp)
    
    def __divmod__(self, other):
        q, _ = np.polydiv(self.coeffs, other.coeffs)
        q = BinaryPolynomial(q, mod_exp=self.mod_exp)
        q.truncate()

        r = self - (q * other)
        r.truncate()
        return q, r
    
    def __mod__(self, other):
        """
        Reduce this polynomial by another polynomial.

        Starting at the leading coefficient of this polynomial, divide by the leading coefficient of the other polynomial.
        This entails subtracting each coefficient by the integer division of the leading coefficient of this polynomial 
        by the leading coefficient of the other polynomial.

        For example,

        129x^2 - 129x % 128x^2 + 128x = x^2 - x
        """
        if self.degree() < other.degree():
            return self
        _, r = divmod(self, other)
        return r
    
    def __call__(self, x):
        return np.polyval(self.coeffs, x) % 2**self.mod_exp
    
    def __repr__(self):
        return np.poly1d(self.coeffs).__repr__()
    
    def __str__(self):
        return np.poly1d(self.coeffs).__str__()
    
    def __eq__(self, other):
        return np.array_equal(self.coeffs, other.coeffs)
    
    def __ne__(self, other):
        return not np.array_equal(self.coeffs, other.coeffs)
    
    def __iter__(self):
        return iter(self.coeffs)

class ZeroIdeal:
    def __init__(self, mod_exp=64):
        """
        Initialize a zero ideal in the ring 2^N.
        """
        self.mod_exp = mod_exp
        self.degree = d_w(self.mod_exp)

        fb = FactorialBasis(self.degree, mod_exp=self.mod_exp)
        self.basis = [BinaryPolynomial([2 ** max(mod_exp-v_2(i), 0)], mod_exp=mod_exp) * fb[i] for i in range(2, self.degree, 2)]

    def __getitem__(self, key):
        return self.basis[key]
    
    def __len__(self):
        return len(self.basis)
    
    def __iter__(self):
        return iter(self.basis)
    
    def __repr__(self):
        return f"ZeroIdeal(2 ^ {self.mod_exp})"
    
    def __str__(self):
        return f"ZeroIdeal(2 ^ {self.mod_exp})"
    
    def print_basis(self):
        for _, basis in enumerate(self.basis):
            print(f"{basis}")

    def simplify(self, poly):
        """
        Simplify a polynomial by subtracting off the zero ideal.
        """
        for z in self[::-1]:
            if z.degree() <= poly.degree():
                poly %= z
        return poly



class FactorialBasis:
    def __init__(self, degree, mod_exp=64):
        """
        Initialize a falling factorial basis in the ring 2^N.
        """
        self.mod_exp = mod_exp
        self.basis = [BinaryPolynomial([1], mod_exp=self.mod_exp)]
        for i in range(0, degree):
            self.basis.append(self.basis[-1] * BinaryPolynomial([1, (-i) % 2**mod_exp], mod_exp=self.mod_exp))

    def __getitem__(self, key):
        return self.basis[key]


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

    H(x) = Sum_{j=0}^{d} h_j x^(j) where x^(j) is the jth falling factorial.

    h_j ~ 0 mod (2 ^ max(w-v_2(j!), 0)) by taking the jth discrete derivative of H(x) and evaluating at 0.

    Note v_2(j!) is the 2-adic valuation of j!.

    """
    poly = np.array([0], dtype=domain(N))
    for i in range(2, d_w(max_degree), 2):
        component_i = random.randint(0, 2**N-1) * G_j(i, N)
        poly = np.polyadd(poly, component_i)
    return poly

def generate_univariate_permutation_polynomial(degree, N=64):
    """
    Let A = Z2n with n ≥ 2 and let 
        
        P (x) = a0 + a1x + · · · + adx^d 
        
        be a polynomial with integral coefficients.
        
        The polynomial P represents a function f of A which is a binary permutation polynomial
        if and only if: 
        - a1 is odd
        - (a2 + a4 + a6 + . . . ) is even
        - (a3 + a5 + a7 + . . . ) is even.
    """
    assert(degree >= 1)
    poly = np.array([0]*(degree+1), dtype=domain(N))
    poly[degree] = 0 # a0 always zero for polynomial in automorphism group_delt 
    poly[degree - 1] = random.randint(1, 2**N-1) | 1 # a1 must be odd

    def _gen_coeffs_even_sum(num_coeffs):
        """
        Generate coefficients which sum to an even number.
        """
        coeffs = [random.randint(0, 2**N-1) for _ in range(num_coeffs)]
        if sum(coeffs) % 2 == 1:
            coeffs[-1] ^= 1 # flip bit of last coefficient to force sum to even
        return coeffs

    evens = _gen_coeffs_even_sum((degree) // 2)     # a2, a4, a6, ...
    assert(sum(evens) % 2 == 0)
    odds = _gen_coeffs_even_sum((degree - 1) // 2)  # a3, a5, a7, ...
    assert(sum(odds) % 2 == 0)

    for i in range(degree - 2, -1, -1):
        poly[i] = odds.pop() if i % 2 else evens.pop()

    assert(len(evens) == 0)
    assert(len(odds) == 0)

    return poly

def derivative(poly):
    """
    Compute the derivative of a polynomial.
    """
    return np.polyder(poly)
        

def univariate_poly_inv(poly, N=64):
    """
    Compute the compositional inverse of a univariate polynomial in the ring 2^N.
    """
    pass

def generate_multivariate_permutation_polynomial(max_degree, num_variables, N=64):
    """
    Generates a multivariate permutation polynomial in the ring 2^N of degree <= max_degree
    with `num_variables` variables.
    """
    pass

def multivariate_poly_inv(poly, N=64):
    """
    Compute the compositional inverse of a multivariate polynomial in the ring 2^N.
    """
    pass

if __name__ == "__main__":
    np.seterr(over='ignore')
    # x = sp.Symbol("x")
    # f = sp.Poly(34*x**3 + 32*x**2 + x, x, domain=sp.GF(256, symmetric=False))
    # f_inv = newton_inverse(f)
    # iden = f_inv.compose(f)
    # print(f"Identity: {iden}")
    # print(iden(5))
    # print(iden(100))
    # t = sp.Poly(128*x**15 + 128*x**13 + 48*x**9 + x, x, modulus=256, symmetric=False)
    # y = t.compose(f)
    # print(y)
    # print(y(5))

    p = BinaryPolynomial([34, 32, 1, 0], mod_exp=8)
    print("\n==Polynomial==\n")
    print(p)

    print("\n==Zero Ideal==\n")
    zero_ideal = ZeroIdeal(mod_exp=8)
    zero_ideal.print_basis()

    print("\n==Inverse (Newton's Method)==\n")
    p_inv = p.newton_inverse(zero_ideal)
    print(p_inv)
    
    print("\n==Verification (Composition should be identity)==\n")
    iden = p.compose(p_inv)
    print(zero_ideal.simplify(iden))
