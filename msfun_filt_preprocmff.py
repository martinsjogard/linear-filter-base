import numpy as np
from scipy.fft import fft, ifft
from msfun_filt_preparecosine import msfun_filt_preparecosine
import mne

def msfun_filt_preprocmff(times, sfreq, cfg):
    """
    Reads and preprocesses signals from an MFF file with selected channels and time samples.

    Parameters:
    - times: array of shape (K, T) or (1, T) with sample times
    - sfreq: sampling frequency (Hz)
    - cfg: dictionary with at least 'mff_file' and 'chans', optional 'filter', 'filt', 'blc'

    Returns:
    - sig: preprocessed signal array
    - cfg: updated configuration
    """
    if not isinstance(times, np.ndarray):
        raise ValueError("times must be a numeric numpy array")
    if not isinstance(sfreq, (int, float)) or sfreq <= 0:
        raise ValueError("sfreq must be a positive scalar")
    if not isinstance(cfg, dict) or 'mff_file' not in cfg or 'chans' not in cfg:
        raise ValueError("cfg must contain 'mff_file' and 'chans'")

    if isinstance(cfg['chans'], str):
        cfg['chans'] = [cfg['chans']]

    if not cfg['mff_file'].endswith('.mff'):
        raise ValueError("cfg.mff_file must end with '.mff'")

    cfg.setdefault('filter', False)
    cfg.setdefault('blc', False)

    if cfg['filter']:
        f = cfg.get('filt', {})
        if not all(k in f for k in ['win', 'par', 'freq', 'width']):
            raise ValueError("Missing required filter parameters in cfg['filt']")
        if not (len(f['par']) == len(f['freq']) == len(f['width'])):
            raise ValueError("Mismatch in lengths of filter parameters")

    # Read raw MFF using MNE (Fieldtrip equivalent)
    print("eeg_preprocess_mff - Reading data (using MNE)...")
    raw = mne.io.read_raw_egi(cfg['mff_file'], preload=True, verbose='ERROR')
    raw.pick_channels(cfg['chans'])
    sig = raw.get_data()
    sfreq = raw.info['sfreq']

    # Time sample selection
    print("msfun_msfun_filt_preprocmff - Getting the right time samples...")
    T = (times * sfreq).astype(int)
    T = T - T.min()
    if times.ndim == 1 or times.shape[0] == 1:
        sig = sig[:, T]
    else:
        K, L = times.shape
        cutsig = np.zeros((K, sig.shape[0], L))
        for k in range(K):
            cutsig[k, :, :] = sig[:, T[k, :]]
        sig = cutsig

    # Filtering
    if cfg['filter']:
        print("msfun_msfun_filt_preprocmff - Filtering data...")
        if sig.ndim == 2:
            win, F = msfun_filt_preparecosine(cfg['filt'], sig.shape[1], sfreq)
            Fsig = fft(sig * win.reshape(1, -1), axis=1)
            sig = np.real(ifft(Fsig * F.reshape(1, -1), axis=1))
        else:
            win, F = msfun_filt_preparecosine(cfg['filt'], sig.shape[2], sfreq)
            for k in range(sig.shape[0]):
                X = sig[k, :, :]
                FX = fft(X * win.reshape(1, -1), axis=1)
                sig[k, :, :] = np.real(ifft(FX * F.reshape(1, -1), axis=1))

    # Baseline correction
    if cfg['blc']:
        print("msfun_msfun_filt_preprocmff - Applying baseline correction...")
        if sig.ndim == 2:
            n = np.where(np.diff((sfreq * times).astype(int)) > 1)[0]
            n = np.concatenate(([0], n, [times.shape[1]]))
            for i in range(len(n) - 1):
                avg = np.mean(sig[:, n[i]+1:n[i+1]], axis=1, keepdims=True)
                sig[:, n[i]+1:n[i+1]] -= avg
        else:
            for k in range(sig.shape[0]):
                avg = np.mean(sig[k, :, :], axis=1, keepdims=True)
                sig[k, :, :] -= avg

    print("msfun_msfun_filt_preprocmff - Data preprocessed and ready.")
    return sig, cfg
