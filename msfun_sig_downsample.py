import numpy as np

def msfun_sig_downsample(sig, cfg):
    """
    Downsamples signal along time axis.

    Parameters:
    - sig: array (C, T) or (K, C, T)
    - cfg: dict with keys:
        - sfreq: original sampling rate (Hz)
        - downsfreq: new sampling rate (Hz)
        - smooth: True to average samples, False to pick samples
        - overlap: (if smooth) number of overlapping buffers

    Returns:
    - sigbis: downsampled signal
    - tsamp: time sample indices in original signal
    """

    if sig is None or cfg is None:
        raise ValueError("sig_downsample requires signal and cfg")

    if not isinstance(sig, np.ndarray) or sig.ndim not in [2, 3]:
        raise ValueError("sig must be 2D or 3D numpy array")

    dim = sig.ndim
    if dim == 2:
        K = 1
        C, T = sig.shape
    else:
        K, C, T = sig.shape

    if not isinstance(cfg, dict) or 'sfreq' not in cfg or 'downsfreq' not in cfg:
        raise ValueError("cfg must be a dict with 'sfreq' and 'downsfreq'")

    sfreq = cfg['sfreq']
    downsfreq = cfg['downsfreq']
    smooth = cfg.get('smooth', True)
    overlap = cfg.get('overlap', 1)

    if not isinstance(sfreq, (int, float)) or sfreq <= 0:
        raise ValueError("cfg.sfreq must be positive")
    if not isinstance(downsfreq, (int, float)) or downsfreq <= 0:
        raise ValueError("cfg.downsfreq must be positive")
    if smooth and (not isinstance(overlap, int) or overlap <= 0 or overlap > T):
        raise ValueError("cfg.overlap must be a positive integer <= T")

    # Calculate buffer size
    N = sfreq / downsfreq
    if not N.is_integer():
        print("sig_downsample - WARNING: sfreq/downsfreq not integer, rounding")
        N = round(N)
        downsfreq = sfreq / N
        print(f"New downsampling frequency: {downsfreq:.2f} Hz")

    N = int(N)
    if overlap >= N:
        overlap = 1

    step = N // overlap
    nsamp = (T - N) // step + 1

    # Time sample indices
    tsamp = np.round(N / 2).astype(int) + np.arange(0, nsamp * step, step)

    # Flatten for ease
    if dim == 3:
        sigbis = sig.transpose(2, 0, 1).reshape(T, K * C).T
    else:
        sigbis = sig

    # Downsampling
    if not smooth:
        sigbis = sigbis[:, tsamp]
    else:
        tbuf = np.arange(0, nsamp * step, step)
        tbuf = np.tile(tbuf, (N, 1)) + np.tile(np.arange(N).reshape(-1, 1), (1, nsamp))
        sigbuf = sigbis[:, tbuf.flatten()]
        sigbuf = sigbuf.reshape(sigbis.shape[0], N, nsamp)
        sigbis = np.mean(sigbuf, axis=1)

    # Restore shape
    if dim == 3:
        sigbis = sigbis.T.reshape(nsamp, K, C).transpose(1, 2, 0)

    return sigbis, tsamp
