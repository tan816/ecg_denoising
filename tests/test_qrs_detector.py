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
