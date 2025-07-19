import numpy as np
from warnings import warn

def sig_spectrum(sig, cfg):
    """
    Compute FFT-based spectrum of a 2D or 3D time series signal.

    Parameters:
    - sig: np.ndarray, shape (C, T) or (K, C, T)
    - cfg: dict with keys:
        - 'sfreq': sampling frequency (Hz)
        - 'type': 'power' or 'fourier' [default 'power']
        - 'average': (only for epoched data) whether to average across epochs

    Returns:
    - Ssig: spectral coefficients or power spectrum
    - freq: corresponding frequency vector
    - band_par: (optional) dictionary of spectrum characteristics
    """

    if sig is None or cfg is None:
        raise ValueError("sig_spectrum requires both signal and configuration")

    if not isinstance(sig, np.ndarray):
        raise TypeError("Signal must be a numeric array")

    if sig.ndim not in [2, 3]:
        raise ValueError("Signal must be 2D or 3D")

    if not isinstance(cfg, dict) or 'sfreq' not in cfg:
        raise ValueError("cfg must include key 'sfreq'")

    sfreq = cfg['sfreq']
    if not isinstance(sfreq, (int, float)) or sfreq <= 0:
        raise ValueError("Sampling frequency must be a positive scalar")

    cfg_type = cfg.get('type', 'power').lower()
    if cfg_type not in ['power', 'fourier']:
        raise ValueError("cfg['type'] must be 'power' or 'fourier'")

    average = bool(cfg.get('average', False)) if sig.ndim == 3 else False

    # Setup frequency vector
    T = sig.shape[-1]
    freq = np.arange(T) * sfreq / T

    # Perform FFT
    Ssig = np.fft.fft(sig, axis=-1)

    # Truncate upper half
    half = T // 2
    freq = freq[:half]
    if sig.ndim == 2:
        Ssig = Ssig[:, :half]
    else:
        Ssig = Ssig[:, :, :half]

    # Power spectrum
    if cfg_type == 'power':
        Ssig = np.abs(Ssig) ** 2

    # Epoch averaging
    if average:
        if cfg_type == 'fourier':
            warn("sig_spectrum - Averaging Fourier coefficients is unusual...")
        Ssig = np.mean(Ssig, axis=0)

    # Spectral summary
    band_par = {}
    if 'return_band_par' in cfg and cfg['return_band_par']:
        P = np.abs(Ssig) ** 2 if cfg_type == 'fourier' else Ssig
        P_sum = np.sum(P, axis=-1, keepdims=True)
        P_norm = P / P_sum
        P_cum = np.cumsum(P_norm, axis=-1)

        abs_diff = lambda x, val: np.abs(x - val)

        band_par['fcenter'] = np.argmin(abs_diff(P_cum, 0.5), axis=-1)
        band_par['nucenter'] = freq[band_par['fcenter']]
        band_par['fmin'] = np.argmin(abs_diff(P_cum, 1e-2), axis=-1)
        band_par['numin'] = freq[band_par['fmin']]
        band_par['fmax'] = np.argmin(abs_diff(P_cum, 1 - 1e-2), axis=-1)
        band_par['numax'] = freq[band_par['fmax']]
        return Ssig, freq, band_par

    return Ssig, freq
