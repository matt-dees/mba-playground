from sympy import *
from sympy.solvers import solve
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.matrices.normalforms import hermite_normal_form
import numpy as np
import ast
from itertools import product



if __name__ == "__main__":
    
    M = DM([[0, 0, 0], [-1, -1, 0], [-1, -1, 0], [-2, 0, -1]], domain=ZZ)
    pprint(M.nullspace())
    
    N = 2
    inputs = list(product(range(2**N), repeat=2))

    funcs = [
        lambda args: args[0] * args[1],
        lambda args: args[0] & args[1],
        lambda args: args[0] | args[1],
        lambda args: args[0] & ~args[1],
        lambda args: ~args[0] & args[1]
    ]

    inp = lambda args: args[0] * args[1]

    A_0 = [lambda args: args[0] & args[1], lambda args: args[0] & ~args[1]]
    A_1 = [lambda args: args[0] | args[1], lambda args: ~args[0] & args[1]]
    

    def f_to_vec(f):
        return np.array(list(map(f, inputs)))
    
    B = f_to_vec(inp).transpose()

    print(np.array(list(map(f_to_vec, A_0))))
    print(np.array(list(map(f_to_vec, A_1))))
    A_0 = np.array(list(map(f_to_vec, A_0))).transpose()
    A_1 = np.array(list(map(f_to_vec, A_1))).transpose()

    print(B)
    np.pprint(A_0)
    print(A_1)

    print(A_0 * A_1)

    print(np.linalg.lstsq(A_0 * A_1, B))
