from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Union


class Matrix:
    """A dense matrix whose entries are stored as :class:`fractions.Fraction`.

    Rationale for using ``Fraction`` instead of ``float``: every supported
    operation (addition, multiplication, determinant, inverse) produces exact
    rational results, so matrices that are known to have integer or simple
    rational entries never accumulate floating-point error. The trade-off is
    speed: arbitrary-precision rationals are dramatically slower than hardware
    floats. This library is for correctness, not throughput.

    Design choice: a single flat list backs the matrix rather than a list of
    lists. This keeps memory locality reasonable for the small matrices the
    library is intended for and makes bounds checks uniform.

    A non-square matrix is a first-class value — only the operations that
    require squareness (``determinant``, ``inverse``) raise ``ValueError``.
    """

    __slots__ = ("_rows", "_cols", "_data")

    def __init__(self, data: Sequence[Sequence[Union[int, float, str, Fraction]]]):
        """Construct a matrix from a sequence of rows.

        Each row is normalised to a :class:`~fractions.Fraction`. ``int`` and
        ``str`` inputs are exact; ``float`` inputs are passed through
        ``Fraction(float)`` so that the *binary* value of the float becomes the
        exact rational — ``Fraction(0.1)`` yields ``3602879701896397/36028797018963968``,
        not ``1/10``. This is a deliberate choice so the conversion is faithful
        to what the caller actually passed, not to what they may have intended
        when writing the literal.
        """
        if isinstance(data, Matrix):
            self._rows = data._rows
            self._cols = data._cols
            self._data = list(data._data)
            return

        rows = len(data)
        if rows == 0:
            cols = 0
        else:
            first = data[0]
            if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
                raise TypeError("Matrix rows must be sequences, got {!r}".format(type(first).__name__))
            cols = len(first)

        flat: List[Fraction] = []
        for r, row in enumerate(data):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError("Row {} must be a sequence, got {!r}".format(r, type(row).__name__))
            if len(row) != cols:
                raise ValueError(
                    "Row {} has {} columns; expected {} (first row width)".format(r, len(row), cols)
                )
            for entry in row:
                if isinstance(entry, Fraction):
                    flat.append(entry)
                else:
                    try:
                        flat.append(Fraction(entry))
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            "Entry at ({}, {}) cannot be converted to Fraction: {!r}".format(r, len(flat) % cols, entry)
                        ) from exc

        self._rows = rows
        self._cols = cols
        self._data = flat

    @property
    def rows(self) -> int:
        """Number of rows (zero for the empty matrix)."""
        return self._rows

    @property
    def cols(self) -> int:
        """Number of columns (zero for the empty matrix)."""
        return self._cols

    def entry(self, i: int, j: int) -> Fraction:
        """Return the entry at row ``i``, column ``j`` (zero-indexed)."""
        return self._data[i * self._cols + j]

    def row(self, i: int) -> List[Fraction]:
        """Return row ``i`` as a list of :class:`~fractions.Fraction`."""
        start = i * self._cols
        return self._data[start : start + self._cols]

    def col(self, j: int) -> List[Fraction]:
        """Return column ``j`` as a list of :class:`~fractions.Fraction`."""
        return [self._data[i * self._cols + j] for i in range(self._rows)]

    def to_list(self) -> List[List[Fraction]]:
        """Return the matrix as a list of lists of :class:`~fractions.Fraction`."""
        return [self.row(i) for i in range(self._rows)]

    def is_square(self) -> bool:
        """True iff the matrix has equal row and column counts."""
        return self._rows == self._cols

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._rows == other._rows and self._cols == other._cols and self._data == other._data

    def __hash__(self) -> int:
        return hash((self._rows, self._cols, tuple(self._data)))

    def __repr__(self) -> str:
        return "Matrix({!r})".format(self.to_list())

    def __getitem__(self, key):
        """Support ``m[i, j]`` for element access.

        Slicing is intentionally unsupported to keep the surface area small
        and the semantics unambiguous.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Use m[i, j] for element access")
        i, j = key
        if not (isinstance(i, int) and isinstance(j, int)):
            raise TypeError("Indices must be integers")
        if not (0 <= i < self._rows and 0 <= j < self._cols):
            raise IndexError("Index ({}, {}) out of bounds for {}x{} matrix".format(i, j, self._rows, self._cols))
        return self._data[i * self._cols + j]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Use m[i, j] = value for element assignment")
        i, j = key
        if not (isinstance(i, int) and isinstance(j, int)):
            raise TypeError("Indices must be integers")
        if not (0 <= i < self._rows and 0 <= j < self._cols):
            raise IndexError("Index ({}, {}) out of bounds for {}x{} matrix".format(i, j, self._rows, self._cols))
        self._data[i * self._cols + j] = value if isinstance(value, Fraction) else Fraction(value)


def add(a: Matrix, b: Matrix) -> Matrix:
    """Element-wise addition. Both matrices must have identical dimensions.

    Raises ``ValueError`` for dimension mismatch (including when either is
    empty and the other is not).
    """
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError(
            "Cannot add {}x{} and {}x{} matrices".format(a.rows, a.cols, b.rows, b.cols)
        )
    data = [x + y for x, y in zip(a._data, b._data)]
    result = Matrix.__new__(Matrix)
    result._rows = a._rows
    result._cols = a._cols
    result._data = data
    return result


def multiply(a: Matrix, b: Matrix) -> Matrix:
    """Matrix product ``A * B``.

    The number of columns of ``a`` must equal the number of rows of ``b``.
    The product of two empty matrices is defined to be empty.
    """
    if a.cols != b.rows:
        raise ValueError(
            "Cannot multiply {}x{} by {}x{}: inner dimensions differ".format(a.rows, a.cols, b.rows, b.cols)
        )
    if a.rows == 0 or b.cols == 0:
        result = Matrix.__new__(Matrix)
        result._rows = a.rows
        result._cols = b.cols
        result._data = []
        return result

    b_cols = b.cols
    a_data = a._data
    b_data = b._data
    flat: List[Fraction] = []
    for i in range(a.rows):
        row_base = i * a.cols
        for j in range(b_cols):
            total = Fraction(0)
            for k in range(a.cols):
                total += a_data[row_base + k] * b_data[k * b_cols + j]
            flat.append(total)
    result = Matrix.__new__(Matrix)
    result._rows = a.rows
    result._cols = b.cols
    result._data = flat
    return result


def determinant(m: Matrix) -> Fraction:
    """Exact determinant via Laplace (cofactor) expansion.

    The base cases ``0x0`` and ``1x1`` are handled directly.

    Complexity is ``O(n!)`` — fine for the small matrices this library is
    designed for. We chose Laplace over LU here for two reasons:

    1. With ``Fraction`` entries, partial pivoting still requires per-step
       comparisons to find a non-zero pivot, which adds branch-heavy code
       for little practical benefit at the intended sizes.
    2. Laplace expansion is straightforward to verify by hand and gives the
       same exact result regardless of pivot choices.
    """
    n = m.rows
    if n != m.cols:
        raise ValueError("Determinant requires a square matrix, got {}x{}".format(m.rows, m.cols))
    if n == 0:
        return Fraction(1)
    if n == 1:
        return m._data[0]
    if n == 2:
        a, b, c, d = m._data
        return a * d - b * c

    data = m._data
    total = Fraction(0)
    sign = 1
    # Expand along the first row. Removing column j of row 0 yields the minor.
    for j in range(n):
        value = data[j]
        if value == 0:
            sign = -sign
            continue
        minor_rows: List[Fraction] = []
        for i in range(1, n):
            base = i * n
            for k in range(n):
                if k == j:
                    continue
                minor_rows.append(data[base + k])
        minor = Matrix.__new__(Matrix)
        minor._rows = n - 1
        minor._cols = n - 1
        minor._data = minor_rows
        total += sign * value * determinant(minor)
        sign = -sign
    return total


def inverse(m: Matrix) -> Matrix:
    """Inverse via the adjugate-cofactor formula.

    ``inverse(A) = adjugate(A) / det(A)`` where the adjugate is the transpose
    of the cofactor matrix. Raises ``ZeroDivisionError`` when the determinant
    is zero (the matrix is singular).

    We compute each cofactor as a single Laplace expansion of the corresponding
    minor, so no separate determinant call is needed for the matrix as a whole
    except the final division check.
    """
    n = m.rows
    if n != m.cols:
        raise ValueError("Inverse requires a square matrix, got {}x{}".format(m.rows, m.cols))
    if n == 0:
        result = Matrix.__new__(Matrix)
        result._rows = 0
        result._cols = 0
        result._data = []
        return result

    det = determinant(m)
    if det == 0:
        raise ZeroDivisionError("Matrix is singular: determinant is zero, no inverse exists")

    data = m._data

    if n == 1:
        result = Matrix.__new__(Matrix)
        result._rows = 1
        result._cols = 1
        result._data = [Fraction(1) / data[0]]
        return result

    cofactors: List[Fraction] = [Fraction(0)] * (n * n)
    for i in range(n):
        for j in range(n):
            minor_rows: List[Fraction] = []
            for r in range(n):
                if r == i:
                    continue
                base = r * n
                for c in range(n):
                    if c == j:
                        continue
                    minor_rows.append(data[base + c])
            minor = Matrix.__new__(Matrix)
            minor._rows = n - 1
            minor._cols = n - 1
            minor._data = minor_rows
            cofactor = determinant(minor)
            if (i + j) % 2 == 1:
                cofactor = -cofactor
            # Transpose on the fly: cofactor of (i, j) lands at (j, i).
            cofactors[j * n + i] = cofactor

    inv_data = [c / det for c in cofactors]
    result = Matrix.__new__(Matrix)
    result._rows = n
    result._cols = n
    result._data = inv_data
    return result
