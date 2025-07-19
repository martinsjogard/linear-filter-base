import numpy as np
from scipy.signal import hilbert

def sig_slow_modulation(Z, fcenter):
    """
    Remove fast oscillation at fcenter from signal Z, keeping only slow modulations.

    Parameters:
    - Z: real or analytic signal, shape (C, T) or (K, C, T)
    - fcenter: frequency index to remove

    Returns:
    - Zslow: signal with fast oscillation removed, keeping slow modulations
    """
    if Z is None or fcenter is None:
        raise ValueError("sig_slow_modulation requires two arguments")

    if not isinstance(Z, np.ndarray) or Z.ndim not in [2, 3]:
        raise ValueError("Z must be a numeric array with 2 or 3 dimensions")

    T = Z.shape[-1]
    is_analytic = np.iscomplexobj(Z)

    if not isinstance(fcenter, int) or fcenter < 1 or fcenter > T:
        raise ValueError("fcenter must be a positive integer index within the time axis length")

    # If real-valued, convert to analytic signal using Hilbert transform
    if not is_analytic:
        Z = hilbert(Z, axis=-1)

    # Construct complex exponential wave to divide out the fast oscillation
    t = np.arange(T)
    wave = np.exp(2j * np.pi * (fcenter - 1) * t / T)

    # Broadcast wave shape to match Z
    shape = [1] * Z.ndim
    shape[-1] = T
    wave = wave.reshape(shape)

    Zslow = Z / wave

    if not is_analytic:
        Zslow = np.real(Zslow)

    return Zslow
