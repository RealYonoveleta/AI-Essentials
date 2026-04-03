from turtle import shape


def get_shape(a):
    """Return the shape of a as a tuple for n-dimensional arrays."""

    shape = []
    while isinstance(a, list):
        shape.append(len(a))
        a = a[0] if a else []

    return tuple(shape)

def get_row(a, i=0):
    """Return the i-th row of a"""

    return a[i]


def get_column(a, j=0):
    """Return the j-th column of a"""

    return [row[j] for row in a]


def dot(a, b):
    """Calculate dot product between a and b"""

    if len(a) != len(b):
        raise ValueError("a and b must have the same size")

    return sum(x * y for x, y in zip(a, b))


def matmul(a, b):
    """Multiply two matrices a and b"""

    if len(a[0]) != len(b):
        raise ValueError("Columns in a must match rows in b")

    aRows = len(a)
    bCols = len(b[0])

    result = [[0] * bCols for _ in range(aRows)]

    for i in range(aRows):
        for j in range(bCols):
            result[i][j] = dot(a[i], get_column(b, j))

    return result
