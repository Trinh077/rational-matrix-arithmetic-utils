import unittest
from fractions import Fraction

from rational_matrix_arithmetic_utils import Matrix, add, multiply, determinant, inverse


class TestMatrixConstruction(unittest.TestCase):
    def test_construction_from_ints(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.entry(0, 0), Fraction(1))
        self.assertEqual(m.entry(1, 1), Fraction(4))

    def test_construction_from_mixed_numeric_types(self):
        m = Matrix([[1, 2.5], ["3/7", Fraction(1, 3)]])
        self.assertEqual(m.entry(0, 1), Fraction(5, 2))
        self.assertEqual(m.entry(1, 0), Fraction(3, 7))
        self.assertEqual(m.entry(1, 1), Fraction(1, 3))

    def test_float_is_binary_value_not_decimal(self):
        """Fraction(0.1) yields the binary value of the float, not 1/10."""
        m = Matrix([[0.1]])
        self.assertEqual(m.entry(0, 0), Fraction(0.1))
        self.assertNotEqual(m.entry(0, 0), Fraction(1, 10))

    def test_ragged_rows_raise(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2, 3], [4, 5]])

    def test_non_numeric_entry_raises_typeerror(self):
        with self.assertRaises((TypeError, ValueError)):
            Matrix([[1, "hello"], [3, 4]])

    def test_getitem_indices(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m[0, 1], Fraction(2))
        self.assertEqual(m[1, 0], Fraction(3))

    def test_getitem_out_of_bounds_raises(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(IndexError):
            m[0, 5]

    def test_is_square(self):
        self.assertTrue(Matrix([[1, 2], [3, 4]]).is_square())
        self.assertFalse(Matrix([[1, 2, 3], [4, 5, 6]]).is_square())

    def test_equality(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 2], [3, 4]])
        c = Matrix([[1, 2], [3, 5]])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def test_copy_via_constructor(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix(a)
        b[0, 0] = 99
        self.assertEqual(a[0, 0], Fraction(1))
        self.assertEqual(b[0, 0], Fraction(99))


class TestAdd(unittest.TestCase):
    def test_add_identity(self):
        a = Matrix([[1, 2], [3, 4]])
        z = Matrix([[0, 0], [0, 0]])
        self.assertEqual(add(a, z), a)

    def test_add_rationals_exact(self):
        a = Matrix([[Fraction(1, 3), Fraction(2, 3)], [Fraction(1, 6), 0]])
        b = Matrix([[Fraction(2, 3), Fraction(1, 3)], [Fraction(5, 6), 1]])
        result = add(a, b)
        self.assertEqual(result, Matrix([[1, 1], [1, 1]]))

    def test_add_dimension_mismatch_raises(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            add(a, b)

    def test_add_1x2_matrices(self):
        a = Matrix([[1, 2]])
        b = Matrix([[3, 4]])
        self.assertEqual(add(a, b), Matrix([[4, 6]]))


class TestMultiply(unittest.TestCase):
    def test_multiply_identity(self):
        a = Matrix([[1, 2], [3, 4]])
        ident = Matrix([[1, 0], [0, 1]])
        self.assertEqual(multiply(a, ident), a)

    def test_multiply_general(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[7, 8], [9, 10], [11, 12]])
        self.assertEqual(multiply(a, b), Matrix([[58, 64], [139, 154]]))

    def test_multiply_dimension_mismatch_raises(self):
        a = Matrix([[1, 2, 3]])
        b = Matrix([[1, 2]])
        with self.assertRaises(ValueError):
            multiply(a, b)

    def test_multiply_result_of_multiplication_is_exact(self):
        a = Matrix([[Fraction(1, 3), Fraction(2, 3)]])
        b = Matrix([[1], [1]])
        self.assertEqual(multiply(a, b), Matrix([[1]]))

    def test_multiply_empty_matrix(self):
        a = Matrix([[]])
        b = Matrix([[]])
        with self.assertRaises(ValueError):
            multiply(a, b)


class TestDeterminant(unittest.TestCase):
    def test_det_1x1(self):
        self.assertEqual(determinant(Matrix([[5]])), Fraction(5))

    def test_det_2x2(self):
        self.assertEqual(determinant(Matrix([[1, 2], [3, 4]])), Fraction(-2))

    def test_det_3x3(self):
        m = Matrix([[2, 0, 1], [0, 3, 0], [1, 0, 2]])
        self.assertEqual(determinant(m), Fraction(9))

    def test_det_4x4(self):
        m = Matrix([
            [1, 0, 2, -1],
            [3, 0, 0, 5],
            [2, 1, 4, -3],
            [1, 0, 5, 0],
        ])
        self.assertEqual(determinant(m), Fraction(30))

    def test_det_singular_returns_zero(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(determinant(m), Fraction(0))

    def test_det_rational_entries_exact(self):
        m = Matrix([[Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 4), Fraction(1, 5)]])
        expected = Fraction(1, 10) - Fraction(1, 12)
        self.assertEqual(determinant(m), expected)

    def test_det_non_square_raises(self):
        with self.assertRaises(ValueError):
            determinant(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_det_0x0_is_one(self):
        self.assertEqual(determinant(Matrix([])), Fraction(1))


class TestInverse(unittest.TestCase):
    def test_inverse_2x2(self):
        m = Matrix([[4, 7], [2, 6]])
        expected = Matrix([[Fraction(6, 10), Fraction(-7, 10)], [Fraction(-2, 10), Fraction(4, 10)]])
        self.assertEqual(inverse(m), expected)

    def test_inverse_times_original_is_identity(self):
        m = Matrix([[4, 7], [2, 6]])
        inv = inverse(m)
        prod = multiply(m, inv)
        self.assertEqual(prod, Matrix([[1, 0], [0, 1]]))

    def test_inverse_rational_times_original_is_identity(self):
        m = Matrix([[Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 4), Fraction(1, 5)]])
        inv = inverse(m)
        prod = multiply(m, inv)
        self.assertEqual(prod, Matrix([[1, 0], [0, 1]]))

    def test_inverse_3x3_times_original_is_identity(self):
        m = Matrix([[2, 0, 1], [0, 3, 0], [1, 0, 2]])
        inv = inverse(m)
        prod = multiply(m, inv)
        self.assertEqual(prod, Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_inverse_singular_raises(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(ZeroDivisionError):
            inverse(m)

    def test_inverse_non_square_raises(self):
        with self.assertRaises(ValueError):
            inverse(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_inverse_1x1(self):
        self.assertEqual(inverse(Matrix([[5]])), Matrix([[Fraction(1, 5)]]))

    def test_inverse_3x3_with_rationals(self):
        m = Matrix([[Fraction(1, 2), 0, 0], [0, Fraction(1, 3), 0], [0, 0, Fraction(1, 4)]])
        inv = inverse(m)
        self.assertEqual(multiply(m, inv), Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


if __name__ == "__main__":
    unittest.main()
