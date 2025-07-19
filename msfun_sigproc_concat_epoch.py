import numpy as np

def msfun_sigproc_concat_epoch(sig, L, type=None):
    """
    Concatenate or epoch a signal array.

    Parameters:
    - sig: np.ndarray of shape (K, N, T) or (N, T)
    - L: number of epochs or epoch length depending on mode
    - type: 'epochnum' or 'epochlength'

    Returns:
    - sigbis: epoched or concatenated signal
    """
    if type is None and sig.ndim == 2:
        type = 'epochlength'
    elif type is None and sig.ndim == 3:
        type = 'epochnum'
    elif type is None or type.lower() not in ['epochnum', 'epochlength']:
        raise ValueError("Type must be 'epochnum' or 'epochlength'")

    if not isinstance(sig, np.ndarray) or sig.ndim not in [2, 3]:
        raise ValueError("sig must be a 2D or 3D numpy array")

    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be a positive integer")

    # Epoching mode
    if sig.ndim == 2:
        N, T = sig.shape
        L = min(L, T)

        if type == 'epochlength':
            epochlength = L
            epochnum = T // L
        elif type == 'epochnum':
            epochnum = L
            epochlength = T // L

        total_len = min(epochnum * epochlength, T)
        sig_trimmed = sig[:, :total_len]
        sigbis = sig_trimmed.reshape(N, epochlength, epochnum)
        sigbis = np.transpose(sigbis, (2, 0, 1))  # (epochnum, N, epochlength)

    # Concatenation mode
    else:
        K, N, T = sig.shape
        L = min(L, K)
        sig_trimmed = sig[:L, :, :]
        sigbis = np.transpose(sig_trimmed, (1, 2, 0))  # (N, T, L)
        sigbis = sigbis.reshape(N, L * T)

    return sigbis
