import permutation_polynomial as pp
import unittest
import numpy as np

class TestZeroGenerator(unittest.TestCase):
    
    def test_degree_2(self):
        np.testing.assert_array_equal(pp.zero_generator(2, 8), np.array([128, 128, 0]))
        for i in range(2**8):
           self.assertEqual(np.polyval(pp.zero_generator(2, 8), i) % 2**8, 0)

    def test_degree_3(self):
        np.testing.assert_array_equal(pp.zero_generator(3, 8), np.array([128, 128, 0, 0]))
        for i in range(2**8):
           self.assertEqual(np.polyval(pp.zero_generator(3, 8), i) % 2**8, 0)
    
    def test_degree_4(self):
        np.testing.assert_array_equal(pp.zero_generator(4, 8), np.array([32, 64, 224, 192, 0]))
        for i in range(2**8):
            self.assertEqual(np.polyval(pp.zero_generator(4, 8), i) % 2**8, 0)
        
    def test_degree_5(self):
        np.testing.assert_array_equal(pp.zero_generator(5, 8), np.array([32, 128, 96, 128, 128, 0]))
        for i in range(2**8):
            self.assertEqual(np.polyval(pp.zero_generator(5, 8), i) % 2**8, 0)

    def test_degree_6(self):
        np.testing.assert_array_equal(pp.zero_generator(6, 8), np.array([16, 80, 112, 240, 0, 64, 0]))
        for i in range(2**8):
            self.assertEqual(np.polyval(pp.zero_generator(6, 8), i) % 2**8, 0)

    def test_null_poly_generator(self):
        N = 8
        for _ in range(100):
            poly = pp.zero_polynomial(6, N)
            for i in range(2**N):
                self.assertEqual(np.polyval(poly, i) % 2**N, 0)
    
    def test_univariate_perm_poly(self):
        N = 8
        for _ in range(100):
            poly_fn = np.poly1d(pp.univariate_permutation_polynomial(6, N))
            vals = {poly_fn(i) % 2**N for i in range(2**N)}
            self.assertEqual(len(vals), 2**N)

        for _ in range(100):
            poly_fn = np.poly1d(pp.univariate_permutation_polynomial(5, N))
            vals = {poly_fn(i) % 2**N for i in range(2**N)}
            self.assertEqual(len(vals), 2**N)

class TestZeroIdealReduction(unittest.TestCase):

    def test_ring_256(self):
        zi = pp.ZeroIdeal(8)
        p = zi.simplify(pp.BinaryPolynomial([129, -129, 0], mod_exp=8))
        self.assertTrue(p == pp.BinaryPolynomial([1, -1, 0], mod_exp=8))
        
class TestNewtonInversion(unittest.TestCase):
    
    def test_inv_ring_256(self):
        zi = pp.ZeroIdeal(8)
        p = pp.BinaryPolynomial([34, 32, 1, 0], mod_exp=8)
        inv = p.newton_inverse(zi)
        self.assertTrue(zi.simplify(p.compose(inv)).is_identity())
        
        for i in range(2**8):
            self.assertEqual(inv(p(i)), i)

if __name__ == '__main__':
    unittest.main()