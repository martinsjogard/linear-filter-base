import numpy as np
from scipy.fft import fft, ifft
from warnings import warn
from copy import deepcopy
from prepare_cosine_filter import prepare_cosine_filter

def sig_filter(sig, cfg):
    """
    Applies a spectral cosine filter to a 2D or 3D signal array.
    """
    if sig is None or cfg is None:
        raise ValueError("sig_filter requires both signal and config")

    if not isinstance(sig, np.ndarray):
        raise TypeError("Signal must be a numeric array")

    if sig.ndim not in [2, 3]:
        raise ValueError("Signal must be 2D or 3D")

    if not isinstance(cfg, dict) or 'sfreq' not in cfg or 'filt' not in cfg:
        raise ValueError("cfg must contain 'sfreq' and 'filt' fields")

    if isinstance(cfg['filt'], str):
        band = cfg['filt'].lower()
        filt_map = {
            'none': None,
            'delta':   {'win': 'boxcar', 'par': ['high', 'low'], 'freq': [1, 4], 'width': [0.5, 0.5]},
            'theta':   {'win': 'boxcar', 'par': ['high', 'low'], 'freq': [4, 8], 'width': [1, 1]},
            'alpha':   {'win': 'boxcar', 'par': ['high', 'low'], 'freq': [8, 12], 'width': [1, 1]},
            'beta':    {'win': 'boxcar', 'par': ['high', 'low'], 'freq': [12, 30], 'width': [2, 2]},
            'betalow': {'win': 'boxcar', 'par': ['high', 'low'], 'freq': [12, 21], 'width': [1, 1]},
            'betahigh':{'win': 'boxcar', 'par': ['high', 'low'], 'freq': [21, 30], 'width': [1, 1]},
            'gamma':   {'win': 'boxcar', 'par': ['high', 'low'], 'freq': [30, 45], 'width': [2, 2]},
            'gammalow':{'win': 'boxcar', 'par': ['high', 'low'], 'freq': [30, 37.5], 'width': [1, 1]},
            'gammahigh':{'win': 'boxcar', 'par': ['high', 'low'], 'freq': [37.5, 45], 'width': [1, 1]}
        }

        if band not in filt_map:
            raise ValueError(f"Unknown filter name: {band}")

        if filt_map[band] is None:
            warn("sig_filter - No filter applied... Just copying data.")
            return deepcopy(sig)
        else:
            cfg['filt'] = filt_map[band]

    filt = cfg['filt']
    if filt is not None:
        if not all(k in filt for k in ['win', 'par', 'freq', 'width']):
            raise ValueError("cfg.filt must contain 'win', 'par', 'freq', 'width'")
        if not (len(filt['par']) == len(filt['freq']) == len(filt['width'])):
            raise ValueError("Mismatch in lengths of 'par', 'freq', and 'width'")

    print("sig_filter - Copying data...")
    sig_filt = deepcopy(sig)

    if filt is not None:
        print("sig_filter - Filtering data...")
        sfreq = cfg['sfreq']

        if sig.ndim == 2:
            n_ch, T = sig.shape
            win, F = prepare_cosine_filter(filt, T, sfreq)
            win = win.reshape(1, -1)
            F = F.reshape(1, -1)
            Fsig = fft(sig * win, axis=1)
            sig_filt = np.real(ifft(Fsig * F, axis=1))
        elif sig.ndim == 3:
            n_epochs, n_ch, T = sig.shape
            win, F = prepare_cosine_filter(filt, T, sfreq)
            for k in range(n_epochs):
                epoch = sig[k, :, :]
                Fsig = fft(epoch * win.reshape(1, -1), axis=1)
                sig_filt[k, :, :] = np.real(ifft(Fsig * F.reshape(1, -1), axis=1))

    print("sig_filter - Filtered data ready.")
    return sig_filt
