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
    
    two_var_signature_vecs = list(product([0, -1], repeat=4))
    
    signature_vec_matrix = Matrix(two_var_signature_vecs, domain=ZZ)
    signature_vec_matrix = signature_vec_matrix.rot90(-1)
    pprint(signature_vec_matrix)
    
    signature_vec_matrix = signature_vec_matrix.row_join(Matrix([42, 42, 42, 42], domain=ZZ))
    pprint(signature_vec_matrix)
    
    print(signature_vec_matrix.nullspace())
