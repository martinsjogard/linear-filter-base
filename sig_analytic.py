import numpy as np
from scipy.fft import fft, ifft

def sig_analytic(X, dim=None):
    """
    Compute the analytic signal of real array X along dimension `dim`.

    Parameters:
    - X: real-valued numpy array
    - dim: dimension along which to compute analytic signal (default = last)

    Returns:
    - Z: complex-valued analytic signal
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if not np.isrealobj(X):
        print("sig_analytic - WARNING: Input is not real. Using real part only.")
        X = np.real(X)

    if dim is None:
        dim = X.ndim - 1

    if not isinstance(dim, int) or dim < 0 or dim >= X.ndim:
        raise ValueError("Invalid dimension")

    if X.shape[dim] == 1:
        raise ValueError("Selected dimension contains only one sample")

    # Move target dimension to front
    X = np.moveaxis(X, dim, 0)
    n = X.shape[0]

    Xf = fft(X, axis=0)
    Zf = np.zeros_like(Xf, dtype=np.complex128)

    if n % 2 == 0:
        Zf[0] = Xf[0]
        Zf[1:n//2] = 2 * Xf[1:n//2]
        Zf[n//2] = Xf[n//2]
    else:
        Zf[0] = Xf[0]
        Zf[1:(n+1)//2] = 2 * Xf[1:(n+1)//2]

    # Inverse FFT to get analytic signal
    Z = ifft(Zf, axis=0)

    # Restore original dimension order
    Z = np.moveaxis(Z, 0, dim)
    return Z
