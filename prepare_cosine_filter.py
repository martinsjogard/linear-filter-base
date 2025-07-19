import numpy as np
from scipy.signal import get_window

def prepare_cosine_filter(opt, T, Fs):
    """
    Compute the window and frequency filter matrix for FFT-based filtering.
    """
    win = get_window(opt['win'], T)  # Create window
    win = win.flatten()              # Ensure row vector

    norm_freqs = np.array(opt['freq']) / Fs * 2
    norm_widths = np.array(opt['width']) / Fs * 2

    F = cos_filt(T, opt['par'], norm_freqs, norm_widths)
    return win, F

def cos_filt(quantum, par_list, f_vect, Ws_vect):
    """
    Generate cosine-shaped frequency filters.
    """
    F_h = np.ones(quantum)

    for k, (f, Ws) in enumerate(zip(f_vect, Ws_vect)):
        flow = quantum * (f - Ws / 2) / 2
        fhigh = quantum * (f + Ws / 2) / 2

        F_mult = np.ones(quantum)

        if par_list[k] == 'low':
            F_mult[:int(np.floor(flow)) + 1] = 1
            F_mult[int(np.ceil(fhigh)) + 1:int(np.ceil((quantum + 1) / 2))] = 0
            trans_idx = np.arange(int(np.ceil(flow)) + 1, int(np.floor(fhigh)) + 2)
            F_mult[trans_idx] = np.cos((trans_idx - flow - 1) / (fhigh - flow) * (np.pi / 2))**2

        elif par_list[k] == 'high':
            F_mult[:int(np.floor(flow)) + 1] = 0
            F_mult[int(np.ceil(fhigh)) + 1:int(np.ceil((quantum + 1) / 2))] = 1
            trans_idx = np.arange(int(np.ceil(flow)) + 1, int(np.floor(fhigh)) + 2)
            F_mult[trans_idx] = np.sin((trans_idx - flow - 1) / (fhigh - flow) * (np.pi / 2))**2

        elif par_list[k] == 'notch':
            F_mult[:int(np.floor(flow)) + 1] = 1
            F_mult[int(np.ceil(fhigh)) + 1:int(np.ceil((quantum + 1) / 2))] = 1
            trans_idx = np.arange(int(np.ceil(flow)) + 1, int(np.floor(fhigh)) + 2)
            F_mult[trans_idx] = np.cos((trans_idx - flow - 1) / (fhigh - flow) * np.pi)**2

        else:
            raise ValueError("The filter must be 'low', 'high', or 'notch'")

        F_mult[int(np.ceil((quantum + 1) / 2)):] = F_mult[int(np.floor((quantum + 1) / 2)) - 1::-1]
        F_h *= F_mult

    return F_h
