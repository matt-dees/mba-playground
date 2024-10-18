import permutation_polynomial as pp
import unittest
import numpy as np

class TestZeroGenerator(unittest.TestCase):
    
    def test_degree_2(self):
        np.testing.assert_array_equal(pp.G_j(2, 8), np.array([128, 128, 0]))
        for i in range(2**8):
           self.assertEqual(np.polyval(pp.G_j(2, 8), i) % 2**8, 0)

    def test_degree_3(self):
        np.testing.assert_array_equal(pp.G_j(3, 8), np.array([128, 128, 0, 0]))
        for i in range(2**8):
           self.assertEqual(np.polyval(pp.G_j(3, 8), i) % 2**8, 0)
    
    def test_degree_4(self):
        np.testing.assert_array_equal(pp.G_j(4, 8), np.array([32, 64, 224, 192, 0]))
        for i in range(2**8):
            self.assertEqual(np.polyval(pp.G_j(4, 8), i) % 2**8, 0)
        
    def test_degree_5(self):
        np.testing.assert_array_equal(pp.G_j(5, 8), np.array([32, 128, 96, 128, 128, 0]))
        for i in range(2**8):
            self.assertEqual(np.polyval(pp.G_j(5, 8), i) % 2**8, 0)

    def test_degree_6(self):
        np.testing.assert_array_equal(pp.G_j(6, 8), np.array([16, 80, 112, 240, 0, 64, 0]))
        for i in range(2**8):
            self.assertEqual(np.polyval(pp.G_j(6, 8), i) % 2**8, 0)

    def test_null_poly_generator(self):
        N = 8
        for _ in range(100):
            poly = pp.generate_zero_polynomial(6, N)
            for i in range(2**N):
                self.assertEqual(np.polyval(poly, i) % 2**N, 0)
    
    def test_univariate_perm_poly(self):
        N = 8
        for _ in range(100):
            poly_fn = np.poly1d(pp.generate_univariate_permutation_polynomial(6, N))
            vals = {poly_fn(i) % 2**N for i in range(2**N)}
            self.assertEqual(len(vals), 2**N)

        for _ in range(100):
            poly_fn = np.poly1d(pp.generate_univariate_permutation_polynomial(5, N))
            vals = {poly_fn(i) % 2**N for i in range(2**N)}
            self.assertEqual(len(vals), 2**N)

if __name__ == '__main__':
    unittest.main()