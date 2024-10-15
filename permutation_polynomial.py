import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix
from sympy.polys.domains import GF


def generate_factorial_basis(N, d):
    """
    Generate a falling factorial basis for the permutation polynomials mod 2^N.
    """
    mtx = sp.matrices.Matrix(domain="GF(256)")
    x = sp.Symbol('x')
    for i in range(0, d + 1):
        if i == 0:
            poly = sp.Poly(1, x, domain='GF(256)').to_ring()
        elif i == 1:
            poly = sp.Poly(x + 0, x, domain='GF(256)').to_ring()
        else:
            poly = poly.mul(sp.Poly(x-(i-1), x, domain='GF(256)').to_ring())
            print(poly)
        mtx = mtx.row_join(sp.Matrix(10, 1, ([0] * (d - i)) + poly.all_coeffs()[::-1]))
        dm = DomainMatrix.from_Matrix(mtx, domain='GF(256)')
        
    return dm

if __name__ == "__main__":
    a = generate_factorial_basis(8, 9)
    print(a)
    sp.pprint(sp.Matrix(a).inv_mod(2**8))