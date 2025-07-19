# linear-filter-base-py

# Signal Processing Toolbox for Electrophysiological Data (Python)

This repository provides a collection of Python functions for preprocessing and analyzing EEG/MEG time-series data. Originally translated from MATLAB, these tools enable flexible and efficient workflows for spectral analysis, signal orthogonalization, leakage correction, filtering, and data preprocessing across various electrophysiological formats.

---

## Overview

The toolbox is designed to support:
- Signal filtering using cosine-tapered frequency domain filters
- Power spectral decomposition
- Hilbert transform and analytic signal extraction
- Signal orthogonalization (statistical and instantaneous)
- Source leakage correction
- Signal windowing, epoching, and downsampling
- Preprocessing signals from MFF and FIFF file formats

These tools are intended for neuroscientific research, especially involving time-frequency analyses and source-space signal interpretation.

---

## Functions

### `msfun_setup_cosine_filter.py`
**Purpose:** Create frequency-domain cosine filter coefficients and the corresponding time-domain taper.  
**Inputs:** 
- `cfg`: Dictionary with keys like `'win'`, `'par'`, `'freq'`, `'width'`
- `N`: Length of the signal
- `sfreq`: Sampling frequency  
**Outputs:** 
- `win`: Time-domain window
- `F`: Frequency-domain filter array

---

### `sig_filter.py`
**Purpose:** Apply frequency-domain cosine filter to 2D or 3D signal.  
**Inputs:** 
- `X`: Signal array (`[chan x time]` or `[epoch x chan x time]`)
- `sfreq`: Sampling frequency
- `cfg`: Filter config dictionary  
**Outputs:** 
- Filtered signal array of same shape

---

### `sig_spectrum.py`
**Purpose:** Compute power spectrum of filtered signal.  
**Inputs:** 
- `X`: Input signal (2D or 3D)
- `sfreq`: Sampling frequency
- `cfg`: Dictionary with filtering and output settings  
**Outputs:** 
- `P`: Power estimate
- `cfg`: Updated config

---

### `sig_slow_modulation.py`
**Purpose:** Extract slow modulation (amplitude or phase) from narrowband analytic signal.  
**Inputs:** 
- `Z`: Complex analytic signal or real-valued narrowband signal
- `fcenter`: Index of center frequency  
**Outputs:** 
- `Zslow`: Signal with slow modulation only

---

### `sig_downsample.py`
**Purpose:** Downsample signal along time axis.  
**Inputs:** 
- `X`: Signal (2D or 3D)
- `r`: Downsampling factor  
**Outputs:** 
- Downsampled signal

---

### `sig_concat_epoch.py`
**Purpose:** Concatenate or epoch signal based on epoch number or length.  
**Inputs:** 
- `sig`: Input signal
- `L`: Number or length of epochs
- `typ`: `'epochnum'` or `'epochlength'`  
**Outputs:** 
- Concatenated or reshaped signal

---

### `sig_inst_orth.py`
**Purpose:** Orthogonalize `X` with respect to `Y` by removing the best linear instantaneous real-valued model.  
**Inputs:** 
- `X`: Complex signal to be corrected
- `Y`: Complex regressor signal  
**Outputs:** 
- `Z`: Orthogonalized signal

---

### `sig_analytic.py`
**Purpose:** Compute analytic signal via Hilbert transform along a specified axis.  
**Inputs:** 
- `X`: Real-valued array
- `axis`: Axis to transform (optional, defaults to last)  
**Outputs:** 
- Complex-valued analytic signal

---

### `sig_leakcorr.py`
**Purpose:** Correct spatial leakage using orthogonalization, regression, or custom coefficients.  
**Inputs:** 
- `X`: Target signal
- `Y`: Source signal (potential leakage)
- `cfg`: Dictionary specifying method: `'orthinst'`, `'orthstat'`, `'custom'`, or `'gcs'`  
**Outputs:** 
- `Z`: Corrected signal

---

### `sig_preprocess_fiff.py`
**Purpose:** Read and preprocess signal from a FIFF file.  
**Inputs:** 
- `raw`: MNE Raw object
- `times`: Time array (continuous or epoched)
- `cfg`: Dictionary including `'chans'`, `'filter'`, `'blc'`, and filter params  
**Outputs:** 
- `sig`: Preprocessed signal
- `cfg`: Updated config dictionary

---

### `sig_preprocess_mff.py`
**Purpose:** Read and preprocess MFF (EGI) recordings using FieldTrip-compatible interface.  
**Inputs:** 
- `times`: Time array
- `sfreq`: Sampling frequency
- `cfg`: Must include `'mff_file'`, `'chans'`, and optionally filter and BLC  
**Outputs:** 
- `sig`: Preprocessed data array
- `cfg`: Updated configuration

---

## Usage

Import the functions as needed in your own scripts or Jupyter notebooks. Example:

```python
from sig_filter import sig_filter
filtered_data = sig_filter(raw_data, sfreq, cfg)
```

## Dependencies
Python 3.8+
NumPy
SciPy
MNE-Python (for sig_preprocess_fiff)
FieldTrip/MAT interface (for sig_preprocess_mff)
