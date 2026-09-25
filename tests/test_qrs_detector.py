import numpy as np
import pytest
from qrs_detector import pan_tompkins_qrs
from data_loader import generate_synthetic_ecg

def test_pan_tompkins_qrs():
    fs = 360
    ecg_data = generate_synthetic_ecg(duration_sec=10.0, fs=fs, condition='NSR')
    clean_sig = ecg_data['signal']
    ground_truth_peaks = ecg_data['ann_sample']
    
    pt_res = pan_tompkins_qrs(clean_sig, fs=fs)
    detected_peaks = pt_res['r_peaks']
    
    assert len(detected_peaks) > 0, "Detector should find peaks in clean signal"
    assert np.all(detected_peaks >= 0) and np.all(detected_peaks < len(clean_sig)), "Peak indices must be within signal bounds"
    
    # Check if detected peaks are close to ground truth
    # Allow a small tolerance (e.g. 50ms)
    tolerance = int(0.050 * fs)
    matched_peaks = 0
    for gt in ground_truth_peaks:
        # find closest detected
        if len(detected_peaks) == 0:
            break
        closest_idx = np.argmin(np.abs(detected_peaks - gt))
        if np.abs(detected_peaks[closest_idx] - gt) <= tolerance:
            matched_peaks += 1
            
    sensitivity = matched_peaks / len(ground_truth_peaks) if len(ground_truth_peaks) > 0 else 0
    assert sensitivity >= 0.90, f"Sensitivity is too low: {sensitivity}"

from qrs_detector import calculate_advanced_hrv_metrics, extract_beat_features

def test_advanced_hrv_metrics():
    # Construct a dummy peak array for 60 seconds (1 peak per second)
    fs = 360
    dummy_peaks = np.arange(0, 60 * fs, fs)
    
    # Perfectly regular -> zero variance -> LF/HF should be handled gracefully
    metrics = calculate_advanced_hrv_metrics(dummy_peaks, fs=fs)
    assert 'lf' in metrics
    assert 'hf' in metrics
    assert 'lf_hf_ratio' in metrics
    assert 'sd1' in metrics
    assert 'sd2' in metrics
    
    # Zero variance check
    assert metrics['lf'] == 0.0
    assert metrics['hf'] == 0.0
    assert np.isnan(metrics['lf_hf_ratio'])
    assert metrics['sd1'] == 0.0
    assert metrics['sd2'] == 0.0
    
    # Short array handling
    short_metrics = calculate_advanced_hrv_metrics(dummy_peaks[:2], fs=fs)
    assert np.isnan(short_metrics['lf'])

def test_extract_beat_features():
    fs = 360
    ecg_data = generate_synthetic_ecg(duration_sec=10.0, fs=fs, condition='NSR')
    pt_res = pan_tompkins_qrs(ecg_data['signal'], fs=fs)
    
    if len(pt_res['r_peaks']) >= 3:
        features, _ = extract_beat_features(ecg_data['signal'], pt_res['r_peaks'], fs=fs)
        assert features.shape[1] == 17, f"Expected 17 features, got {features.shape[1]}"
