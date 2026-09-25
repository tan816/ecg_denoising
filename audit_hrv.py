import numpy as np
import pandas as pd
from data_loader import load_mitbih_record
from qrs_detector import pan_tompkins_qrs, calculate_advanced_hrv_metrics
from ml_classifier import create_cardiac_dataset, CLASS_NAMES

def audit_synthetic_classes():
    print("--- 1. ADVANCED HRV FEATURE SANITY CHECKS (SYNTHETIC) ---")
    X_raw, y_raw, _, X_filt, y_filt, _, _ = create_cardiac_dataset(n_samples_per_class=10, add_noise=False)
    
    # We will use X_filt (clean data basically, since add_noise=False)
    # The advanced features are at indices:
    # 12: LF, 13: HF, 14: LF/HF, 15: SD1, 16: SD2
    
    feature_names = ['LF', 'HF', 'LF/HF', 'SD1', 'SD2']
    indices = [12, 13, 14, 15, 16]
    
    for cls_idx, class_name in enumerate(CLASS_NAMES):
        print(f"\nClass: {class_name}")
        mask = (y_filt == cls_idx)
        if not np.any(mask):
            print("  No samples found.")
            continue
            
        X_class = X_filt[mask]
        
        for name, idx in zip(feature_names, indices):
            vals = X_class[:, idx]
            # filter out nans for stats
            vals_valid = vals[~np.isnan(vals)]
            if len(vals_valid) == 0:
                print(f"  {name:6s}: All NaN")
            else:
                mean_v = np.mean(vals_valid)
                std_v = np.std(vals_valid)
                min_v = np.min(vals_valid)
                max_v = np.max(vals_valid)
                print(f"  {name:6s}: Mean = {mean_v:8.3f}, Std = {std_v:8.3f}, Min = {min_v:8.3f}, Max = {max_v:8.3f}")

def check_mitbih():
    print("\n--- 2. REAL-DATA COMPATIBILITY CHECK (MIT-BIH) ---")
    try:
        import wfdb
    except ImportError:
        print("wfdb not installed. Installing wfdb to fetch real MIT-BIH record...")
        import subprocess
        subprocess.run(["pip", "install", "wfdb"], check=True)
        import wfdb

    record = load_mitbih_record('100', sampto=3600)  # 10 seconds
    sig = record['signal']
    fs = record['fs']
    
    print(f"Loaded MIT-BIH Record {record['record_name']} - Length: {len(sig)} samples at {fs} Hz")
    
    pt_res = pan_tompkins_qrs(sig, fs=fs)
    r_peaks = pt_res['r_peaks']
    
    print(f"Detected R-peaks: {len(r_peaks)}")
    if len(r_peaks) > 1:
        rr_ms = np.diff(r_peaks) / fs * 1000.0
        print(f"Valid RR intervals found: {len(rr_ms)}")
    else:
        print("Not enough R-peaks to compute RR intervals.")
        return
        
    metrics = calculate_advanced_hrv_metrics(r_peaks, fs=fs)
    print("HRV Metrics Calculated:")
    for k, v in metrics.items():
        if k != 'poincare_coords':
            print(f"  {k:15s}: {v}")
    
    nans = sum(1 for v in metrics.values() if isinstance(v, float) and np.isnan(v))
    print(f"Total NaN values in advanced HRV features: {nans}")

if __name__ == '__main__':
    audit_synthetic_classes()
    check_mitbih()
