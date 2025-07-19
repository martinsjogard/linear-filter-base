import numpy as np

def msfun_sig_leakcorr(X, Y, cfg):
    """
    Apply leakage correction from signal Y to signal X.

    Parameters:
    - X: array (N, T)
    - Y: array (M, T)
    - cfg: dictionary with keys:
        - method: 'gcs', 'orthinst', 'orthstat', or 'custom'
        - cfg.gcs: for 'gcs', contains 'inv' and 'ind'
        - cfg.beta: for 'custom', shape (N, M)

    Returns:
    - Z: corrected signal (N, T)
    """

    if not (isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)):
        raise TypeError("X and Y must be numpy arrays")

    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2D arrays with matching time dimension")

    method = cfg.get('method', '').lower()
    if method not in ['gcs', 'orthinst', 'orthstat', 'custom']:
        raise ValueError("cfg['method'] must be one of 'gcs', 'orthinst', 'orthstat', 'custom'")

    if method == 'gcs':
        if Y.shape[0] != 1:
            raise ValueError("GCS method requires Y to have shape (1, T)")
        gcs = cfg.get('gcs', {})
        inv = gcs.get('inv', {})
        ind = gcs.get('ind', None)

        if not (isinstance(ind, int) and ind >= 1 and
                'nsource' in inv and 'invop' in inv and 'leadfield' in inv and
                inv['invop'].shape[0] == inv['nsource'] and
                ind <= inv['nsource']):
            raise ValueError("Missing or invalid GCS configuration")

        beta = inv['invop'] @ inv['leadfield'][:, ind - 1]
        beta = beta / beta[ind - 1]
        Z = X - np.outer(beta, Y[0])

    elif method == 'orthstat':
        beta = (X.real @ Y.real.T) @ np.linalg.pinv(Y.real @ Y.real.T)
        Z = X - beta @ Y

    elif method == 'orthinst':
        if np.isrealobj(X) and np.isrealobj(Y):
            raise ValueError("X and Y must be complex for 'orthinst'")
        ratio = np.sum(Y**2, axis=0) / np.sum(np.abs(Y)**2, axis=0)
        ratio = ratio[np.newaxis, :]  # broadcast shape
        Z = 0.5 * (X - np.conj(X) * ratio)

    elif method == 'custom':
        beta = cfg.get('beta')
        if not isinstance(beta, np.ndarray) or beta.shape != (X.shape[0], Y.shape[0]):
            raise ValueError("cfg['beta'] must be array of shape (N, M)")
        Z = X - beta @ Y

    return Z
