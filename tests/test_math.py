from ai_essentials.math import *

def test_shape():
    """Test the get_shape function."""
    a = [[1, 2], [3, 4]]
    assert get_shape(a) == (2, 2)
    
def test_row():    
    """Test the get_row function."""
    a = [[1, 2], [3, 4]]
    assert get_row(a, 0) == [1, 2]
    assert get_row(a, 1) == [3, 4]

def test_column():
    """Test the get_column function."""
    a = [[1, 2], [3, 4]]
    assert get_column(a, 0) == [1, 3]
    assert get_column(a, 1) == [2, 4] 
    
def test_dot():
    """Test the dot function."""
    a = [1, 2]
    b = [3, 4]
    assert dot(a, b) == 11
    
def test_matmul():
    """Test the matmul function."""
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    assert matmul(a, b) == expected
    