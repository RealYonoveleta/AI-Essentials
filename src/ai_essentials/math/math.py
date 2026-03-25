def get_shape(a):
    """Return the shape of a as a tuple (rows, columns)"""

    return len(a), len(a[0]) if a else 0

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
    aCols_bRows = len(a[0])
    bCols = len(b[0])

    result = [[0] * bCols for _ in range(aRows)]

    for i in range(aRows):
        for j in range(bCols):
            sum = 0
            for t in range(aCols_bRows):
                sum += a[i][t] * b[t][j]

            result[i][j] = sum

    return result
