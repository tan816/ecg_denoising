"""
QRS Complex & Peak Detection and Feature Extraction Engine (Stage 3)
Implements the Pan-Tompkins Algorithm for robust R-peak detection and extracts
clinically relevant time-domain (RR, PR, QT, QRS width, HR, HRV) and morphological features.
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, Any, List, Tuple


def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: float = 360.0) -> Dict[str, Any]:
    """
    Implements the classical Pan-Tompkins QRS Detection Algorithm.
    
    Steps:
    1. Bandpass Filtering (5-15 Hz)
    2. Derivative Filter (Slope highlighting)
    3. Squaring Function (Non-linear amplification)
    4. Moving Window Integration (Pulse width smoothing)
    5. Dynamic Adaptive Thresholding for R-Peak detection
    """
    N = len(ecg_signal)
    nyq = 0.5 * fs
    
    # 1. Bandpass Filter (5 - 15 Hz)
    lowcut = 5.0 / nyq
    highcut = 15.0 / nyq
    b_bp, a_bp = signal.butter(1, [lowcut, highcut], btype='bandpass')
    bandpassed = signal.filtfilt(b_bp, a_bp, ecg_signal)
    
    # 2. Derivative Filter: y[n] = 1/8 (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])
    der_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (fs / 8.0)
    differentiated = np.convolve(bandpassed, der_kernel, mode='same')
    
    # 3. Squaring Function
    squared = differentiated ** 2
    
    # 4. Moving Window Integrator (Window size ~ 150 ms)
    win_samples = int(0.150 * fs)
    kernel = np.ones(win_samples) / win_samples
    integrated = np.convolve(squared, kernel, mode='same')
    
    # 5. Adaptive Thresholding
    # Find local maxima in integrated signal
    min_distance = int(0.200 * fs)  # Refractory period: 200ms
    peaks_mwi, _ = signal.find_peaks(integrated, distance=min_distance, height=np.mean(integrated))
    
    # Refine R-peaks by matching local maxima in filtered/raw ECG within a search window
    r_peaks = []
    search_win = int(0.080 * fs)  # 80 ms search window around MWI peak
    
    for p in peaks_mwi:
        win_start = max(0, p - search_win)
        win_end = min(N, p + search_win)
        local_max = win_start + np.argmax(ecg_signal[win_start:win_end])
        
        # Guard against duplicate or extremely close peaks
        if len(r_peaks) == 0 or (local_max - r_peaks[-1]) >= min_distance:
            r_peaks.append(local_max)
            
    r_peaks = np.array(r_peaks, dtype=int)
    
    return {
        'r_peaks': r_peaks,
        'bandpassed': bandpassed,
        'differentiated': differentiated,
        'squared': squared,
        'integrated': integrated
    }


def calculate_hrv_metrics(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """
    Calculates standard Heart Rate Variability (HRV) metrics:
    - SDNN: Standard deviation of NN (RR) intervals
    - RMSSD: Root mean square of successive RR interval differences
    - pNN50: Percentage of successive RR interval differences > 50 ms
    """
    if len(rr_intervals_ms) < 2:
        return {'sdnn': 0.0, 'rmssd': 0.0, 'pnn50': 0.0}

    sdnn = float(np.std(rr_intervals_ms))
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0
    
    nn50 = np.sum(np.abs(successive_diffs) > 50.0)
    pnn50 = float((nn50 / len(successive_diffs)) * 100.0) if len(successive_diffs) > 0 else 0.0
    
    return {
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50
    }


def calculate_advanced_hrv_metrics(r_peaks: np.ndarray, fs: float = 360.0) -> Dict[str, Any]:
    """
    Calculates advanced HRV metrics including Frequency-Domain (LF, HF, LF/HF)
    using the Lomb-Scargle periodogram (appropriate for unevenly sampled RR series),
    and Non-Linear Poincaré features (SD1, SD2, coordinates).
    """
    if len(r_peaks) < 4:
        return {
            'lf': np.nan, 'hf': np.nan, 'lf_hf_ratio': np.nan,
            'sd1': np.nan, 'sd2': np.nan,
            'poincare_coords': ([], [])
        }

    # RR series
    rr_sec = np.diff(r_peaks) / fs
    rr_ms = rr_sec * 1000.0
    t_rr = r_peaks[1:] / fs  # Time of occurrence of each RR interval

    # 1. Frequency Domain (Lomb-Scargle)
    # Define frequencies: 0.01 to 0.5 Hz
    f = np.linspace(0.01, 0.5, 1000)
    w = 2 * np.pi * f
    # Subtract mean before periodogram
    rr_centered = rr_ms - np.mean(rr_ms)
    
    # Check if signal has zero variance (e.g., constant synthetic RR)
    if np.var(rr_centered) < 1e-6:
        lf, hf, lf_hf = 0.0, 0.0, np.nan
    else:
        pgram = signal.lombscargle(t_rr, rr_centered, w, normalize=False)
        # Scale to PSD (approximation)
        df = f[1] - f[0]
        
        lf_mask = (f >= 0.04) & (f < 0.15)
        hf_mask = (f >= 0.15) & (f < 0.40)
        
        lf = float(np.sum(pgram[lf_mask]) * df)
        hf = float(np.sum(pgram[hf_mask]) * df)
        lf_hf = float(lf / hf) if hf > 0 else np.nan

    # 2. Poincaré Non-linear Analysis
    rr_n = rr_ms[:-1]
    rr_n1 = rr_ms[1:]
    
    if len(rr_n) < 2:
        sd1, sd2 = np.nan, np.nan
    else:
        sd1 = float(np.sqrt(0.5 * np.std(rr_n1 - rr_n)**2))
        sd2 = float(np.sqrt(0.5 * np.std(rr_n1 + rr_n)**2))

    return {
        'lf': lf,
        'hf': hf,
        'lf_hf_ratio': lf_hf,
        'sd1': sd1,
        'sd2': sd2,
        'poincare_coords': (rr_n.tolist(), rr_n1.tolist())
    }


def extract_beat_features(ecg_signal: np.ndarray, r_peaks: np.ndarray, fs: float = 360.0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts comprehensive clinical and morphological feature vectors per heartbeat.
    
    Features extracted per beat:
    1. Preceding RR interval (RR_pre in ms)
    2. Following RR interval (RR_post in ms)
    3. RR Ratio (RR_pre / mean_RR)
    4. Instantaneous Heart Rate (BPM)
    5. QRS Width (ms)
    6. Estimated PR Interval (ms)
    7. Estimated QT Interval (ms)
    8. R-peak Amplitude (mV)
    9. Q-wave Depth (mV)
    10. S-wave Depth (mV)
    11. T-peak Amplitude (mV)
    12. LF Power (Global segment feature)
    13. HF Power (Global segment feature)
    14. LF/HF Ratio (Global segment feature)
    15. Poincaré SD1 (Global segment feature)
    16. Poincaré SD2 (Global segment feature)
    """
    N = len(ecg_signal)
    n_peaks = len(r_peaks)
    
    if n_peaks < 3:
        raise ValueError(f"Insufficient R-peaks detected ({n_peaks}) to compute features.")
        
    rr_intervals_sec = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals_sec)
    
    adv_hrv = calculate_advanced_hrv_metrics(r_peaks, fs=fs)
    
    feature_matrix = []
    clinical_records = []
    
    for i in range(1, n_peaks - 1):
        r_curr = r_peaks[i]
        
        # RR intervals
        rr_pre = (r_peaks[i] - r_peaks[i-1]) / fs * 1000.0   # ms
        rr_post = (r_peaks[i+1] - r_peaks[i]) / fs * 1000.0  # ms
        rr_ratio = rr_pre / (mean_rr * 1000.0)
        instant_hr = 60.0 / (rr_pre / 1000.0)
        
        # Segment window around R-peak (e.g. [-150ms, +350ms])
        w_start = max(0, r_curr - int(0.150 * fs))
        w_end = min(N, r_curr + int(0.350 * fs))
        beat_segment = ecg_signal[w_start:w_end]
        
        # Locate Q wave (min before R-peak)
        q_search_start = max(0, r_curr - int(0.060 * fs))
        q_idx = q_search_start + np.argmin(ecg_signal[q_search_start:r_curr+1])
        q_amp = ecg_signal[q_idx]
        
        # Locate S wave (min after R-peak)
        s_search_end = min(N, r_curr + int(0.080 * fs))
        s_idx = r_curr + np.argmin(ecg_signal[r_curr:s_search_end])
        s_amp = ecg_signal[s_idx]
        
        # QRS Duration (Q-index to S-index)
        qrs_duration = (s_idx - q_idx) / fs * 1000.0  # ms
        if qrs_duration <= 0 or qrs_duration > 300:
            qrs_duration = np.nan
            
        # Estimate P-wave peak (max in [-220ms, -80ms] before R)
        p_start = max(0, r_curr - int(0.240 * fs))
        p_end = max(0, r_curr - int(0.080 * fs))
        if p_end > p_start:
            p_idx = p_start + np.argmax(ecg_signal[p_start:p_end])
            pr_interval = (r_curr - p_idx) / fs * 1000.0
            p_amp = ecg_signal[p_idx]
        else:
            pr_interval = np.nan
            p_amp = np.nan
            
        # Estimate T-wave peak (max in [+150ms, +350ms] after R)
        t_start = min(N, r_curr + int(0.150 * fs))
        t_end = min(N, r_curr + int(0.350 * fs))
        if t_end > t_start:
            t_idx = t_start + np.argmax(ecg_signal[t_start:t_end])
            qt_interval = (t_idx - q_idx) / fs * 1000.0
            t_amp = ecg_signal[t_idx]
        else:
            qt_interval = np.nan
            t_amp = np.nan
            
        r_amp = ecg_signal[r_curr]
        
        # Feature Vector
        feat_vector = [
            rr_pre,
            rr_post,
            rr_ratio,
            instant_hr,
            qrs_duration,
            pr_interval,
            qt_interval,
            r_amp,
            q_amp,
            s_amp,
            p_amp,
            t_amp,
            adv_hrv['lf'],
            adv_hrv['hf'],
            adv_hrv['lf_hf_ratio'],
            adv_hrv['sd1'],
            adv_hrv['sd2']
        ]
        feature_matrix.append(feat_vector)
        
        clinical_records.append({
            'r_peak_sample': r_curr,
            'time_sec': r_curr / fs,
            'hr_bpm': instant_hr,
            'rr_pre_ms': rr_pre,
            'qrs_width_ms': qrs_duration,
            'pr_interval_ms': pr_interval,
            'qt_interval_ms': qt_interval
        })
        
    return np.array(feature_matrix), clinical_records
