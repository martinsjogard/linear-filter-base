import numpy as np

def sig_inst_orth(X, Y):
    """
    Instantaneously orthogonalize signal X with respect to Y.

    Parameters:
    - X: complex-valued array of shape (N, T)
    - Y: complex-valued array of shape (M, T)

    Returns:
    - Z: orthogonalized signal of shape (N, T)
    """

    if X is None or Y is None:
        raise ValueError("sig_inst_orth requires two input arrays")

    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays")

    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("Input arrays must be 2D and have the same number of time samples")

    if np.isrealobj(X) and np.isrealobj(Y):
        raise ValueError("X and Y must be complex-valued to perform orthogonalization")

    # Compute Z using Hipp et al. (2012)-style formula
    norm_ratio = np.sum(Y**2, axis=0) / np.sum(np.abs(Y)**2, axis=0)  # shape: (T,)
    norm_ratio = norm_ratio[np.newaxis, :]  # broadcast to (1, T)

    proj = np.conj(X) * norm_ratio  # shape: (N, T)
    Z = 0.5 * (X - proj)  # shape: (N, T)

    return Z
