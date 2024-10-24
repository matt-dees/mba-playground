import numpy as np
import sympy as sp
import math
import random
import functools

class BinaryPolynomial:
    
    def __init__(self, coeffs, mod_exp=64):
        """
        Initialize a polynomial with coefficients `coeffs` in the ring 2^N.

        Coefficients are stored with leading coefficient at the top of the array.
        """
        self.mod_exp = mod_exp
        self.coeffs = np.array([int(x) % 2 ** mod_exp for x in coeffs], 
                               dtype=numpy_dtype_from_exp(mod_exp))

    @staticmethod
    def polyx(mod_exp=64):
        """
        Return the polynomial f(x) = x in the ring 2^N.
        """
        return BinaryPolynomial([1, 0], mod_exp=mod_exp)
    
    @staticmethod
    def constant(c, mod_exp=64):
        """
        Return the polynomial f(x) = C in the ring 2^N.
        """
        return BinaryPolynomial([c], mod_exp=mod_exp)
    
    def degree(self):
        """
        Return the degree of the polynomial.
        """
        return len(self.coeffs) - 1
    
    def derivative(self):
        """
        Compute the derivative of the polynomial.

        The derivative of a polynomial is the sum of the derivatives of each term.

        d/dx (a_n x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0) = n a_n x^{n-1} + (n-1) a_{n-1} x^{n-2} + ... + a_1

        """
        return BinaryPolynomial(np.polyder(self.coeffs), mod_exp=self.mod_exp)
    
    def compose(self, other):
        """
        Compute the composition of this polynomial with another polynomial.

        f . g = f(g(x))

        """
        return BinaryPolynomial(np.polyval(np.poly1d(self.coeffs), np.poly1d(other.coeffs)), mod_exp=self.mod_exp)
    
    def truncate(self):
        """
        Remove leading zeros from the polynomial.

        0x^3 + x^2 + 0x + 0 -> x^2 + 0x + 0

        [0 1 0 0] -> [1 0 0]

        """
        self.coeffs = np.trim_zeros(self.coeffs, 'f')

    def is_identity(self):
        """
        Compare this polynomial to the identity polynomial x (polyx) in the ring 2^N.
        """
        return np.array_equal(BinaryPolynomial.polyx(self.mod_exp).coeffs, self.coeffs)

    def newton_inverse(self, zero_ideal):
        """
        Compute the Newton inverse of a polynomial in the ring 2^N.

        The algorithm is inspired by the original Newton-Raphson method for finding roots of
        a polynomial, with the key reduction step being:

            x_{n+1} = x_n - f(x_n) / f'(x_n)

        Instead of roots, we are looking for an inverse. Meaning, instead of solving for f(x) = 0,
        we are solving for f(g(x)) = x. Thus, the reduction step becomes:

            g_{n+1} = g_n - f(g_n - X) / f'(g_n)

        In the ring 2^N division is not defined for all ring values, as there are zero divisors (any even number).

        To avoid division, we can rewrite the division by f'(g_n) as multiplication by the modular inverse of f'(g_n).

        We can easily calculate this because:

            f . g          = x
            (f . g)'       = 1
            (f' . g ) * g' = 1     (by Leibniz rule)
            g'             = 1 / f . g'

        Thus, the multiplicative inverse of f'(g_n) is the derivative of g_n (aka g').

        The reduction step then becomes:
            
                g_{n+1} = g_n - g_n' * f(g_n - X)

        See:

        Barhelemy, Lucas, et al. "Binary permutation polynomial inversion and application to obfuscation techniques." 
        Proceedings of the 2016 ACM Workshop on Software PROtection. 2016.

        for details.
        """

        # Initial guess is f(^-1)(x) = x
        g_i = BinaryPolynomial.polyx(self.mod_exp)
        iter_count = 0

        while True: 
            assert(iter_count < self.degree()*3)

            # Current candidate is g_i
            # Compute f . g_i to see if it is the identity function.
            # If so, we're done.
            composition = zero_ideal.simplify(self.compose(g_i))
            if composition.is_identity():
                break

            # Not the correct inverse yet. Perform Newton reduction step
            # adjusted for inverse finding.
            #
            # g_{n+1} = g_n - g_n' * (f . g_n - x)
            g_i -= g_i.derivative() * (composition - BinaryPolynomial.polyx(self.mod_exp))
            g_i = zero_ideal.simplify(g_i)

            iter_count += 1
            
        return g_i

    def __add__(self, other):
        """
        Add two polynomials together.

        The sum of two polynomials is the sum of the coefficients of each term.

        (a_n x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0) + (b_n x^n + b_{n-1} x^{n-1} + ... + b_1 x + b_0) = 
        
        (a_n + b_n) x^n + (a_{n-1} + b_{n-1}) x^{n-1} + ... + (a_1 + b_1) x + (a_0 + b_0)
        """
        return BinaryPolynomial(np.polyadd(self.coeffs, other.coeffs), mod_exp=self.mod_exp)
    
    def __sub__(self, other):
        """
        Subtract two polynomials.
        
        The difference of two polynomials is the difference of the coefficients of each term.

        (a_n x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0) - (b_n x^n + b_{n-1} x^{n-1} + ... + b_1 x + b_0) = 

        (a_n - b_n) x^n + (a_{n-1} - b_{n-1}) x^{n-1} + ... + (a_1 - b_1) x + (a_0 - b_0)
        
        """
        return BinaryPolynomial(np.polysub(self.coeffs, other.coeffs), mod_exp=self.mod_exp)
    
    def __mul__(self, other):
        """
        Multiply two polynomials.

        The product of two polynomials is the convolution of the coefficients of each term.

        (a_n x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0) * (b_m x^m + b_{m-1} x^{m-1} + ... + b_1 x + b_0) =

        (a_n b_m) x^{n+m} + (a_n b_{m-1} + a_{n-1} b_m) x^{n+m-1} + ... + (a_1 b_0 + a_0 b_1) x + (a_0 b_0)
        
        """
        return BinaryPolynomial(np.polymul(self.coeffs, other.coeffs), mod_exp=self.mod_exp)
    
    def __divmod__(self, other):
        """
        Divide this polynomial by another polynomial.

        Starting at the leading coefficient of this polynomial, divide by the leading coefficient of the other polynomial.

        Each coefficients is `reduced` by the integer division of the leading coefficient of this polynomial by the leading coefficient of the other polynomial.

        Return the quotient and the remainder. (q, r)

        For example,

        129x^2 - 129x // 128x^2 + 128x = (1, x(x - 1))
        """
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
        """
        Evaluate the polynomial at x using the notation f(x).
        
        p = BinaryPolynomial([1, 2, 3])
        p(5) -> 1 + 2*5 + 3*5^2 -> 1 + 10 + 75 -> 86
        """
        return np.polyval(self.coeffs, x) % 2**self.mod_exp
    
    def __repr__(self):
        """
        Overload of the __repr__ method to print the polynomial in a more readable format.
        """
        return np.poly1d(self.coeffs).__repr__()
    
    def __str__(self):
        """
        Overload of the __str__ method to print the polynomial in a more readable format.
        """
        return np.poly1d(self.coeffs).__str__()
    
    def __eq__(self, other):
        """
        Check if two polynomials are equal.
        
        Two polynomials are equal iff. their coefficients are equal.
        """
        return np.array_equal(self.coeffs, other.coeffs)
    
    def __ne__(self, other):
        """
        Check if two polynomials are not equal.

        Two polynomials are not equal iff. their coefficients are not _all_ equal.
        """
        return not np.array_equal(self.coeffs, other.coeffs)
    
    def __iter__(self):
        """
        Overload __iter__ method to iterate over the coefficients of the polynomial.
        """
        return iter(self.coeffs)

class ZeroIdeal:
    def __init__(self, mod_exp=64):
        """
        Initialize a zero ideal in the ring 2^N.

        The zero ideal is the set of polynomials that are divisible by the zero ideal basis of degree `d`.

        These polynomials are of the form:

           H(x) = Sum_{j=0}^{d} h_j x^(j) where x^(j) is the jth falling factorial.

           h_j ~ 0 mod (2 ^ max(w-v_2(j!), 0)) by taking the jth discrete derivative of H(x) and evaluating at 0.

        """
        self.mod_exp = mod_exp
        self.degree = poly_degree_in_ring(self.mod_exp)

        fb = FactorialBasis(self.degree, mod_exp=self.mod_exp)
        self.basis = [BinaryPolynomial([2 ** max(mod_exp-p_adic_valuation(2, i), 0)], mod_exp=mod_exp) * fb[i] for i in range(2, self.degree, 2)]

    def __getitem__(self, key):
        """
        Get the `key`th polynomial in the zero ideal basis.
        
        Warning: Not necessarily degree `key`.
        """
        return self.basis[key]
    
    def __len__(self):
        """
        Return the number of polynomials in the zero ideal basis.
        """
        return len(self.basis)
    
    def __iter__(self):
        """
        Overload __iter__ method to iterate over the polynomials in the zero ideal basis.
        """
        return iter(self.basis)
    
    def __repr__(self):
        return f"ZeroIdeal(2 ^ {self.mod_exp})"
    
    def __str__(self):
        """
        Overload __str__ method to print the zero ideal basis one polynomial at a time.
        """
        return functools.reduce(lambda x, y: f"{x}\n{y}", self.basis)

    def simplify(self, poly):
        """
        Simplify a polynomial by mapping it to the residue class of the zero ideal basis.
        """
        for z in self[::-1]:
            if z.degree() <= poly.degree():
                poly %= z
        return poly



class FactorialBasis:
    def __init__(self, degree, mod_exp=64):
        """
        Initialize a falling factorial basis in the ring 2^N.

        The falling factorial basis is a set of polynomials of the form x(x-1)(x-2)...(x-d) mod 2^N.

        [0] = 1
        [1] = x
        [2] = x(x-1) = x^2 - x
        [3] = x(x-1)(x-2) = x^3 - 3x^2 + 2x
        ...

        It's easy to see that this forms a basis for Z2^n[x] as the determinant of the matrix formed by the coefficients
        is 1 (triangular matrix with 1s on the diagonal).
        """
        self.mod_exp = mod_exp

        # Initialize the basis with the constant polynomial f(x) = 1
        self.basis = [BinaryPolynomial.constant(1, mod_exp=self.mod_exp)]

        # Generate the falling factorial basis:
        #
        #       f(x) = x(x-1)(x-2)...(x-d) = f(x-1) * (x-d)
        #
        # by multiplying the previous polynomial by (x - i) for i = 1, 2, ..., d
        for i in range(0, degree):
            self.basis.append(self.basis[-1] * BinaryPolynomial([1, -i], mod_exp=self.mod_exp))

    def __getitem__(self, key):
        return self.basis[key]
    
def numpy_dtype_from_exp(exp):
    """
    Return the NumPy data type that can hold values in the ring 2^exp.
    """
    match exp:
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

def poly_degree_in_ring(w=64):
    """
    Computes the degree of the largest unique polynomial in the ring 2^N.

    Find the smallest N such that the 2-adic valuation of N! is greater than or equal to w.
    """
    curr = 1
    N = 1
    while not curr >= w:
        N += 1
        curr = p_adic_valuation(2, N)
    return N

def p_adic_valuation(p, n):
    """
    Computes the p-adic valuation of n! using Legendre's formula.
    """
    return sum([math.floor(n / p**i) for i in range(1, math.floor(math.log(n, p)) + 1)])

def falling_factorial_basis(degree, mod_exp=64):
    """
    Generate a falling factorial basis of degree `d` for the permutation polynomials mod 2^N.

    This implementation directly relies on the underlying storage of the polynomial coefficients in NumPy
    to compute mod 2^N.
    """

    dtype = numpy_dtype_from_exp(mod_exp)
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
    Compute the modular inverse of an _upper_ triangular matrix in the ring 2^N from the
    extended NxN Heisenberg group.

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

def zero_generator(j, N=64):
    """
    Computes the zero polynomial generator for degree j in the ring 2^N.
    """
    basis = falling_factorial_basis(j, N)
    c_j = 2 ** max(N-p_adic_valuation(2, j), 0)
    ret = c_j * basis[:,j]
    return np.flip(ret)

def zero_polynomial(max_degree, N=64):
    """
    Generates a zero polynomial in the ring 2^N of degree <= max_degree by
    choosing random coefficients to multiply by the zero ideal basis of degree `max_degree`.

    H(x) = Sum_{j=0}^{d} h_j x^(j) where x^(j) is the jth falling factorial.

    h_j ~ 0 mod (2 ^ max(w-v_2(j!), 0)) by taking the jth discrete derivative of H(x) and evaluating at 0.

    Note v_2(j!) is the 2-adic valuation of j!.

    """

    poly = np.array([0], dtype=numpy_dtype_from_exp(N))
    for i in range(2, poly_degree_in_ring(max_degree), 2):
        component_i = random.randint(0, 2**N-1) * zero_generator(i, N)
        poly = np.polyadd(poly, component_i)
    return poly

def univariate_permutation_polynomial(degree, N=64):
    """
    Let A = Z2n with n ≥ 2 and let 
        
        P (x) = a0 + a1x + · · · + adx^d 
        
        The polynomial P represents a function f of A which is a binary permutation polynomial
        if and only if: 
        - a1 is odd
        - (a2 + a4 + a6 + . . . ) is even
        - (a3 + a5 + a7 + . . . ) is even.
        
    Generate these polynomials by choosing random coefficients for the polynomial and ensuring criteria is met.
    If not met, simply change the parity of one of the terms.
    """
    assert(degree >= 1)
    poly = np.array([0]*(degree+1), dtype=numpy_dtype_from_exp(N))
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

def multivariate_permutation_polynomial(max_degree, num_variables, N=64):
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

    p = BinaryPolynomial([34, 32, 1, 0], mod_exp=8)
    print("\n==Polynomial==\n")
    print(p)

    print("\n==Zero Ideal==\n")
    zero_ideal = ZeroIdeal(mod_exp=8)
    print(zero_ideal)

    print("\n==Inverse (Newton's Method)==\n")
    p_inv = p.newton_inverse(zero_ideal)
    print(p_inv)
    
    print("\n==Verification (Composition should be identity)==\n")
    iden = p.compose(p_inv)
    print(zero_ideal.simplify(iden))
