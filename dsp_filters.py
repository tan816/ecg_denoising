"""
DSP Filtering Engine (Stage 2)
Implements classical (IIR Butterworth Highpass/Lowpass, Notch) and modern (Discrete Wavelet Transform)
denoising algorithms for ECG noise removal (Baseline Wander, Powerline, EMG, Electrode Motion).
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, Any, Tuple

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def highpass_filter(data: np.ndarray, fs: float = 360.0, cutoff: float = 0.5, order: int = 2) -> np.ndarray:
    """
    Butterworth High-pass Filter for Baseline Wander Removal (< 0.5 Hz).
    Uses zero-phase forward-backward filtering (filtfilt).
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data)


def notch_filter(data: np.ndarray, fs: float = 360.0, notch_freq: float = 50.0, quality_factor: float = 30.0) -> np.ndarray:
    """
    IIR Notch Filter for Powerline Interference Removal (50 Hz / 60 Hz).
    """
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = signal.iirnotch(w0, quality_factor)
    return signal.filtfilt(b, a, data)


def lowpass_filter(data: np.ndarray, fs: float = 360.0, cutoff: float = 45.0, order: int = 4) -> np.ndarray:
    """
    Butterworth Low-pass Filter for High-Frequency EMG Artifact Removal (> 45 Hz).
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def wavelet_denoise(data: np.ndarray, 
                    wavelet: str = 'db6', 
                    level: int = 5, 
                    threshold_mode: str = 'soft') -> np.ndarray:
    """
    Modern Discrete Wavelet Transform (DWT) Denoising using Universal Thresholding (VisuShrink).
    
    Decomposes signal into approximation and detail coefficients, estimates noise variance
    from Median Absolute Deviation (MAD) of detail level 1, applies soft thresholding, and reconstructs.
    """
    if not HAS_PYWT:
        print("Warning: PyWavelets (pywt) not installed. Returning original data.")
        return data

    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Estimate noise level sigma from MAD of first detail coefficients (high frequency)
    detail_1 = coeffs[-1]
    sigma = np.median(np.abs(detail_1 - np.median(detail_1))) / 0.6745
    
    # Donoho-Johnstone Universal Threshold
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    # Apply soft thresholding to detail coefficients (keep approximation coeff intact)
    denoised_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=threshold_mode))
        
    # Reconstruct signal
    return pywt.waverec(denoised_coeffs, wavelet)[:len(data)]


def apply_dsp_pipeline(ecg_signal: np.ndarray, 
                       fs: float = 360.0, 
                       notch_freq: float = 50.0, 
                       use_wavelet: bool = True) -> Dict[str, np.ndarray]:
    """
    Executes complete classical + modern DSP filtering cascade on raw/noisy ECG.
    
    Cascade Sequence:
    1. High-pass (0.5 Hz) -> Remove baseline wander
    2. Notch (50/60 Hz) -> Suppress powerline interference
    3. Low-pass (45 Hz) -> Filter muscle noise
    4. Wavelet Denoising -> Remove residual non-stationary artifacts
    """
    # Step 1: Baseline Wander Removal
    hp_filtered = highpass_filter(ecg_signal, fs=fs, cutoff=0.5, order=2)
    
    # Step 2: Powerline Interference Removal
    notch_filtered = notch_filter(hp_filtered, fs=fs, notch_freq=notch_freq, quality_factor=30.0)
    
    # Step 3: Low-pass EMG Removal
    lp_filtered = lowpass_filter(notch_filtered, fs=fs, cutoff=45.0, order=4)
    
    # Step 4: Modern Wavelet Denoising
    if use_wavelet and HAS_PYWT:
        final_filtered = wavelet_denoise(lp_filtered, wavelet='db6', level=5, threshold_mode='soft')
    else:
        final_filtered = lp_filtered

    return {
        'raw_signal': ecg_signal,
        'hp_filtered': hp_filtered,
        'notch_filtered': notch_filtered,
        'lp_filtered': lp_filtered,
        'final_filtered': final_filtered
    }
