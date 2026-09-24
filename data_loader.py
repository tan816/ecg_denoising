"""
Data Acquisition & Synthetic Generation Module (Stage 1)
Supports fetching real annotated MIT-BIH Arrhythmia records via PhysioNet WFDB
and generating realistic synthetic multi-condition ECG signals with injected noise artifacts.
"""

import os
import numpy as np
import scipy.signal as signal
import pandas as pd
from typing import Tuple, Dict, Any, Optional

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False


def load_mitbih_record(record_name: str = '100', 
                       sampto: int = 3600, 
                       pn_dir: str = 'mitdb') -> Dict[str, Any]:
    """
    Fetch a real record from the PhysioNet MIT-BIH Arrhythmia Database.
    
    Parameters:
        record_name: Record ID (e.g., '100', '105', '106', '119', '200', '203')
        sampto: Number of samples to read (default 3600 = 10s at 360Hz)
        pn_dir: PhysioNet database directory ('mitdb')
        
    Returns:
        Dict containing signal array, sampling frequency fs, channel names, and annotations.
    """
    if not HAS_WFDB:
        raise ImportError("wfdb package is not installed. Install with `pip install wfdb`.")
        
    try:
        record = wfdb.rdrecord(record_name, sampto=sampto, pn_dir=pn_dir)
        annotation = wfdb.rdann(record_name, 'atr', sampto=sampto, pn_dir=pn_dir)
        
        fs = record.fs
        lead_ii = record.p_signal[:, 0]  # Usually MLII lead
        
        return {
            'record_name': record_name,
            'signal': lead_ii,
            'fs': fs,
            'units': record.units[0] if record.units else 'mV',
            'comments': record.comments,
            'ann_sample': annotation.sample,
            'ann_symbol': annotation.symbol,
            'source': 'MIT-BIH Database'
        }
    except Exception as e:
        print(f"Warning: Could not fetch MIT-BIH record {record_name} from PhysioNet: {e}")
        print("Falling back to synthetic ECG signal generation.")
        return generate_synthetic_ecg(duration_sec=sampto/360.0, fs=360, condition='NSR')


def generate_single_beat(fs: int = 360, 
                         qrs_width_ms: float = 80.0, 
                         is_pvc: bool = False,
                         has_p_wave: bool = True) -> np.ndarray:
    """
    Synthesizes a single P-QRS-T ECG heartbeat template using Gaussian combination.
    """
    dt = 1.0 / fs
    # Standard beat period ~ 800ms
    beat_duration = 0.8  # seconds
    t = np.arange(-0.3, beat_duration - 0.3, dt)
    ecg_beat = np.zeros_like(t)

    if is_pvc:
        # PVC: No P-wave, wide QRS (>120ms), inverted T wave, larger amplitude
        qrs_w = qrs_width_ms / 1000.0  # e.g., 140ms
        # Wide QRS peak
        ecg_beat += 1.4 * np.exp(-((t - 0.0) ** 2) / (2 * (qrs_w / 2.355) ** 2))
        # Deep S wave
        ecg_beat -= 0.6 * np.exp(-((t - 0.05) ** 2) / (2 * 0.02 ** 2))
        # Large inverted T wave
        ecg_beat -= 0.45 * np.exp(-((t - 0.35) ** 2) / (2 * 0.06 ** 2))
    else:
        # Normal morphology
        # P wave
        if has_p_wave:
            ecg_beat += 0.18 * np.exp(-((t + 0.16) ** 2) / (2 * 0.022 ** 2))
        # Q wave
        ecg_beat -= 0.15 * np.exp(-((t + 0.025) ** 2) / (2 * 0.008 ** 2))
        # R peak
        qrs_w = qrs_width_ms / 1000.0
        ecg_beat += 1.1 * np.exp(-((t - 0.0) ** 2) / (2 * (qrs_w / 2.355) ** 2))
        # S wave
        ecg_beat -= 0.25 * np.exp(-((t - 0.035) ** 2) / (2 * 0.01 ** 2))
        # T wave
        ecg_beat += 0.32 * np.exp(-((t - 0.28) ** 2) / (2 * 0.045 ** 2))

    return t, ecg_beat


def generate_synthetic_ecg(duration_sec: float = 10.0, 
                           fs: int = 360, 
                           heart_rate: float = 72.0, 
                           condition: str = 'NSR',
                           random_seed: int = 42) -> Dict[str, Any]:
    """
    Generates realistic multi-condition synthetic ECG signals with physiological ground truth.
    
    Supported Conditions:
        - 'NSR': Normal Sinus Rhythm (60-100 BPM, regular RR)
        - 'Bradycardia': Slow Heart Rate (< 60 BPM)
        - 'Tachycardia': Fast Heart Rate (> 100 BPM)
        - 'AFib': Atrial Fibrillation (Irregular RR intervals, absent/noisy P waves)
        - 'PVC': Premature Ventricular Contraction (Premature ectopic beats with wide QRS)
    """
    np.random.seed(random_seed)
    n_samples = int(duration_sec * fs)
    ecg_clean = np.zeros(n_samples)
    r_peaks = []
    beat_labels = []

    # Target heart rates
    if condition == 'Bradycardia':
        target_hr = 48.0
    elif condition == 'Tachycardia':
        target_hr = 130.0
    else:
        target_hr = heart_rate

    mean_rr = 60.0 / target_hr
    current_time = 0.3  # Start offset

    while current_time < duration_sec - 0.5:
        # Determine RR interval and beat type based on condition
        is_pvc = False
        has_p_wave = True
        label = 'N'

        if condition == 'AFib':
            # Irregular RR interval variation (+/- 35%)
            rr = mean_rr * np.random.uniform(0.65, 1.35)
            has_p_wave = False  # Absent P wave in AFib
            label = 'AF'
        elif condition == 'PVC':
            # Every 4th beat is a PVC
            if len(r_peaks) > 0 and (len(r_peaks) + 1) % 4 == 0:
                is_pvc = True
                rr = mean_rr * 0.65  # Premature beat
                has_p_wave = False
                label = 'V'
            elif len(r_peaks) > 0 and beat_labels[-1] == 'V':
                rr = mean_rr * 1.35  # Full compensatory pause
                label = 'N'
            else:
                rr = mean_rr * np.random.normal(1.0, 0.03)
                label = 'N'
        elif condition == 'Bradycardia':
            rr = mean_rr * np.random.normal(1.0, 0.02)
            label = 'B'
        elif condition == 'Tachycardia':
            rr = mean_rr * np.random.normal(1.0, 0.02)
            label = 'T'
        else:  # NSR
            rr = mean_rr * np.random.normal(1.0, 0.03)
            label = 'N'

        r_sample = int(current_time * fs)
        if r_sample >= n_samples:
            break

        r_peaks.append(r_sample)
        beat_labels.append(label)

        # Generate beat waveform
        qrs_w = 140.0 if is_pvc else 80.0
        _, beat_wave = generate_single_beat(fs=fs, qrs_width_ms=qrs_w, is_pvc=is_pvc, has_p_wave=has_p_wave)

        # Overlay beat wave into signal buffer
        start_idx = r_sample - int(0.3 * fs)
        end_idx = start_idx + len(beat_wave)

        if start_idx >= 0 and end_idx < n_samples:
            ecg_clean[start_idx:end_idx] += beat_wave
        elif start_idx < 0:
            valid_len = len(beat_wave) + start_idx
            ecg_clean[0:end_idx] += beat_wave[-valid_len:]
        elif end_idx >= n_samples:
            valid_len = n_samples - start_idx
            ecg_clean[start_idx:n_samples] += beat_wave[:valid_len]

        current_time += rr

    # If AFib, add baseline fibrillatory waves (f-waves: 4-9 Hz low amplitude noise)
    if condition == 'AFib':
        t_arr = np.linspace(0, duration_sec, n_samples)
        f_waves = 0.05 * np.sin(2 * np.pi * 6.5 * t_arr) + 0.03 * np.cos(2 * np.pi * 8.2 * t_arr)
        ecg_clean += f_waves

    return {
        'record_name': f'Synthetic_{condition}',
        'signal': ecg_clean,
        'fs': fs,
        'units': 'mV',
        'condition': condition,
        'ann_sample': np.array(r_peaks),
        'ann_symbol': beat_labels,
        'source': 'Synthetic Engine'
    }


def inject_noise(clean_signal: np.ndarray,
                 fs: int = 360,
                 baseline_wander_amp: float = 0.45,
                 bw_freq: float = 0.15,
                 powerline_amp: float = 0.25,
                 pl_freq: float = 50.0,
                 emg_amp: float = 0.18,
                 electrode_motion_amp: float = 0.30,
                 random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Injects realistic biomedical and instrumentation noise components into a clean ECG signal.
    
    Noise Types:
    1. Baseline Wander: Low frequency sinusoidal respiration artifact (< 0.5 Hz).
    2. Powerline Interference: 50 Hz (or 60 Hz) sinusoidal grid hum + harmonics.
    3. EMG Artifacts: High frequency muscle activity noise (> 35 Hz).
    4. Electrode Motion (EM): Random baseline shift / drift.
    """
    np.random.seed(random_seed)
    N = len(clean_signal)
    t = np.arange(N) / fs
    
    # 1. Baseline Wander (Respiration / Body Movement)
    baseline_wander = baseline_wander_amp * np.sin(2 * np.pi * bw_freq * t + 0.4)
    baseline_wander += (baseline_wander_amp * 0.3) * np.cos(2 * np.pi * (bw_freq * 2.2) * t)

    # 2. Powerline Interference (50 Hz grid + 3rd harmonic)
    powerline = powerline_amp * np.sin(2 * np.pi * pl_freq * t)
    powerline += (powerline_amp * 0.2) * np.sin(2 * np.pi * (pl_freq * 3) * t)

    # 3. EMG Artifacts (Bandpass high-frequency muscle noise 35-100 Hz)
    raw_gaussian = np.random.randn(N)
    b_emg, a_emg = signal.butter(4, [35.0 / (fs / 2.0), 100.0 / (fs / 2.0)], btype='bandpass')
    emg_noise = emg_amp * signal.filtfilt(b_emg, a_emg, raw_gaussian)

    # 4. Electrode Motion (Random walk shift)
    em_steps = np.random.randn(N)
    b_em, a_em = signal.butter(2, 0.8 / (fs / 2.0), btype='lowpass')
    electrode_motion = electrode_motion_amp * signal.filtfilt(b_em, a_em, em_steps)

    # Total combined noise
    total_noise = baseline_wander + powerline + emg_noise + electrode_motion
    noisy_signal = clean_signal + total_noise

    return {
        'noisy_signal': noisy_signal,
        'clean_signal': clean_signal,
        'total_noise': total_noise,
        'baseline_wander': baseline_wander,
        'powerline': powerline,
        'emg_noise': emg_noise,
        'electrode_motion': electrode_motion
    }


def compute_snr(clean_signal: np.ndarray, noisy_or_filtered_signal: np.ndarray) -> float:
    """
    Computes Signal-to-Noise Ratio (SNR) in decibels (dB).
    SNR_dB = 10 * log10( P_signal / P_noise )
    """
    noise = clean_signal - noisy_or_filtered_signal
    p_signal = np.sum(clean_signal ** 2)
    p_noise = np.sum(noise ** 2)
    
    if p_noise == 0 or np.isnan(p_noise):
        return 100.0  # Infinite / perfect SNR
        
    return 10.0 * np.log10(p_signal / p_noise)


def compute_mse(clean_signal: np.ndarray, noisy_or_filtered_signal: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE)."""
    return float(np.mean((clean_signal - noisy_or_filtered_signal) ** 2))


def compute_psnr(clean_signal: np.ndarray, noisy_or_filtered_signal: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = compute_mse(clean_signal, noisy_or_filtered_signal)
    if mse == 0:
        return 100.0
    max_val = np.max(np.abs(clean_signal))
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))
