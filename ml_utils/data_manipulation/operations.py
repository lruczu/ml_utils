import numpy as np 


def repeat_across_axis(arr: np.ndarray, dim: int, axis: int) -> np.ndarray:
    """
    parameters
    ----------
    arr: its shape is [d0, d1, .., dn-1]
    dim: >= 1
    axis: one of 0, 1, 2, .., n
    
    As an example, let's consider 'arr' as 3-dimensional array with shape [n, m, k]
    In this case the function can work in four modes:
    
    - repeat_across_axis(arr, dim, axis=0) -> np.ndarray of shape [dim, n, m, k]
    - repeat_across_axis(arr, dim, axis=1) -> np.ndarray of shape [n, dim, m, k]
    - repeat_across_axis(arr, dim, axis=2) -> np.ndarray of shape [n, m, dim, k]
    - repeat_across_axis(arr, dim, axis=3) -> np.ndarray of shape [n, m, k, dim]
    """
    n = len(arr.shape)
    if axis > n:
        raise ValueError('axis must be an integer between 0 and {}'.format(n))
    return np.repeat(np.expand_dims(arr, axis=axis), dim, axis=axis)
