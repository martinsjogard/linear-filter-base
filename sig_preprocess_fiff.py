import numpy as np
from scipy.fft import fft, ifft
from mne.io import Raw
from prepare_cosine_filter import prepare_cosine_filter

def sig_preprocess_fiff(raw: Raw, times, cfg):
    """
    Reads and preprocesses signals from an MNE Raw object with selected channels and time samples.

    Parameters:
    - raw: MNE Raw object
    - times: array of shape (K, T) or (1, T) with sample times
    - cfg: dictionary with at least 'chans' and optional 'filter', 'filt', and 'blc'

    Returns:
    - sig: processed signal array
    - cfg: updated configuration
    """

    if raw is None or times is None or cfg is None:
        raise ValueError("sig_preprocess_fiff requires 3 inputs")

    if 'chans' not in cfg:
        raise ValueError("cfg must contain a list of channels as 'chans'")

    if isinstance(cfg['chans'], str):
        cfg['chans'] = [cfg['chans']]

    cfg.setdefault('filter', False)
    cfg.setdefault('blc', False)

    if cfg['filter']:
        f = cfg.get('filt', {})
        if not all(k in f for k in ['win', 'par', 'freq', 'width']):
            raise ValueError("Missing required filter parameters in cfg['filt']")
        if not (len(f['par']) == len(f['freq']) == len(f['width'])):
            raise ValueError("Mismatch in lengths of filter parameters")

    # Match channel names
    all_chs = raw.info['ch_names']
    chan_inds = [all_chs.index(ch) for ch in cfg['chans'] if ch in all_chs]
    cfg['signal'] = {
        'chan': chan_inds,
        'ind': [i for i, ch in enumerate(cfg['chans']) if ch in all_chs],
        'names': [cfg['chans'][i] for i in range(len(cfg['chans'])) if cfg['chans'][i] in all_chs]
    }

    if not cfg['signal']['chan']:
        print("sig_preprocess_fiff - WARNING: No channels read... Returning empty output.")
        return np.array([]), cfg

    # Read raw data for selected channels
    print("sig_preprocess_fiff - Reading data...")
    data, _ = raw[cfg['signal']['chan'], :]
    sig = data

    # Select correct time samples
    print("sig_preprocess_fiff - Getting the right time samples...")
    times = np.asarray(times)
    T = (times * raw.info['sfreq']).astype(int) - raw.first_samp
    if T.ndim == 1:
        sig = sig[:, T]
    else:
        K, L = T.shape
        sig_epo = np.zeros((K, len(cfg['signal']['chan']), L))
        for k in range(K):
            sig_epo[k, :, :] = sig[:, T[k, :]]
        sig = sig_epo

    # Filter if requested
    if cfg['filter']:
        print("sig_preprocess_fiff - Filtering data...")
        sfreq = raw.info['sfreq']
        if sig.ndim == 2:
            win, F = prepare_cosine_filter(cfg['filt'], sig.shape[1], sfreq)
            win = win.reshape(1, -1)
            F = F.reshape(1, -1)
            Fsig = fft(sig * win, axis=1)
            sig = np.real(ifft(Fsig * F, axis=1))
        else:
            win, F = prepare_cosine_filter(cfg['filt'], sig.shape[2], sfreq)
            for k in range(sig.shape[0]):
                X = sig[k, :, :]
                FX = fft(X * win.reshape(1, -1), axis=1)
                sig[k, :, :] = np.real(ifft(FX * F.reshape(1, -1), axis=1))

    # Baseline correction
    if cfg['blc']:
        print("sig_preprocess_fiff - Applying baseline correction...")
        if sig.ndim == 2:
            n = np.where(np.diff((raw.info['sfreq'] * times).astype(int)) > 1)[0]
            n = np.concatenate(([0], n, [times.shape[1]]))
            for i in range(len(n) - 1):
                avg = np.mean(sig[:, n[i]:n[i+1]], axis=1, keepdims=True)
                sig[:, n[i]:n[i+1]] -= avg
        else:
            for k in range(sig.shape[0]):
                avg = np.mean(sig[k, :, :], axis=1, keepdims=True)
                sig[k, :, :] -= avg

    print("sig_preprocess_fiff - Data preprocessed and ready.")
    return sig, cfg
