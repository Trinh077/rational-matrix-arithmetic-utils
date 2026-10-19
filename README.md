# rational-matrix-arithmetic-utils

Exact rational matrix arithmetic for Python's standard library. Matrix addition, multiplication, determinant, and inverse are computed with `fractions.Fraction` entries so that integer and simple-rational matrices never accumulate floating-point error.

## What it exports

- `Matrix` — a matrix class whose entries are `Fraction`.
- `add(a, b)` — element-wise sum of two matrices.
- `multiply(a, b)` — matrix product.
- `determinant(m)` — exact determinant.
- `inverse(m)` — exact inverse.

## Usage

```python
from fractions import Fraction
from rational_matrix_arithmetic_utils import Matrix, add, multiply, determinant, inverse

a = Matrix([[1, 2], [3, 4]])
b = Matrix([[5, 6], [7, 8]])

print(add(a, b).to_list())       # [[6, 8], [10, 12]]
print(multiply(a, b).to_list())  # [[19, 22], [43, 50]]
print(determinant(a))            # -2
print(inverse(a))                # Matrix([[-2, 1], [3/2, -1/2]])
```

## Why this exists

Numerical linear algebra over the rationals is useful when a result must be checked against an exact value or fed back into further exact arithmetic. Floats introduce rounding error that compounds across operations; `Fraction` does not, at the cost of being far slower. This library targets correctness and small matrices, not throughput.

## The awkward edge

`float` literals passed to `Matrix` are converted via `Fraction(float)`, which captures the binary value the float actually represents — not the decimal the source text suggests. `Matrix([[0.1]])` holds `3602879701896397/36028797018963968`, not `1/10`. If you want `1/10`, pass the string `"1/10"` or `Fraction(1, 10)`.

Singular matrices passed to `inverse` raise `ZeroDivisionError`; non-square matrices passed to `determinant` or `inverse` raise `ValueError`.

Determinant and inverse use Laplace (cofactor) expansion, which is `O(n!)`. This is fine for the intended use case of small matrices and keeps the implementation easy to audit.
