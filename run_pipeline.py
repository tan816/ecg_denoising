import os
import pickle
import numpy as np
import pandas as pd
from ml_classifier import run_full_ml_benchmark, CLASS_NAMES
from data_loader import generate_synthetic_ecg, inject_noise, compute_snr, compute_mse, compute_psnr
from dsp_filters import apply_dsp_pipeline
from qrs_detector import pan_tompkins_qrs


def evaluate_dsp_noise_levels():
    print("-------------------------------------------------------------------------")
    print(" 1. CONTROLLED DSP BENCHMARK (Synthetic ECG + Noise Injection)")
    print("-------------------------------------------------------------------------")
    
    noise_profiles = {
        'Low': {'bw': 0.1, 'pl': 0.1, 'emg': 0.05, 'em': 0.05},
        'Medium': {'bw': 0.45, 'pl': 0.25, 'emg': 0.18, 'em': 0.30},
        'High': {'bw': 0.8, 'pl': 0.6, 'emg': 0.4, 'em': 0.7}
    }
    
    fs = 360
    ecg_data = generate_synthetic_ecg(duration_sec=30.0, fs=fs, condition='NSR', random_seed=42)
    clean_sig = ecg_data['signal']
    
    results = []
    for level, params in noise_profiles.items():
        noisy_res = inject_noise(
            clean_sig, fs=fs, random_seed=42,
            baseline_wander_amp=params['bw'], powerline_amp=params['pl'],
            emg_amp=params['emg'], electrode_motion_amp=params['em']
        )
        noisy_sig = noisy_res['noisy_signal']
        dsp_res = apply_dsp_pipeline(noisy_sig, fs=fs, use_wavelet=True)
        filtered_sig = dsp_res['final_filtered']
        
        snr_in = compute_snr(clean_sig, noisy_sig)
        snr_out = compute_snr(clean_sig, filtered_sig)
        mse_val = compute_mse(clean_sig, filtered_sig)
        psnr_val = compute_psnr(clean_sig, filtered_sig)
        
        results.append({
            'Noise Level': level,
            'Input SNR (dB)': f"{snr_in:.2f}",
            'Output SNR (dB)': f"{snr_out:.2f}",
            'Improvement (dB)': f"+{snr_out - snr_in:.2f}",
            'MSE': f"{mse_val:.4f}",
            'PSNR (dB)': f"{psnr_val:.2f}"
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("\n")


def evaluate_pan_tompkins():
    print("-------------------------------------------------------------------------")
    print(" 2. PAN-TOMPKINS QRS DETECTION VALIDATION")
    print("-------------------------------------------------------------------------")
    fs = 360
    ecg_data = generate_synthetic_ecg(duration_sec=60.0, fs=fs, condition='NSR', random_seed=99)
    clean_sig = ecg_data['signal']
    ground_truth = ecg_data['ann_sample']
    
    pt_res = pan_tompkins_qrs(clean_sig, fs=fs)
    detected = pt_res['r_peaks']
    
    tolerance = int(0.050 * fs) # 50 ms
    tp = 0
    for gt in ground_truth:
        if len(detected) == 0: break
        idx = np.argmin(np.abs(detected - gt))
        if np.abs(detected[idx] - gt) <= tolerance:
            tp += 1
            
    fp = len(detected) - tp
    fn = len(ground_truth) - tp
    sensitivity = tp / len(ground_truth) if len(ground_truth) > 0 else 0
    ppv = tp / len(detected) if len(detected) > 0 else 0
    
    rr_true = np.mean(np.diff(ground_truth)) / fs * 1000
    rr_det = np.mean(np.diff(detected)) / fs * 1000 if len(detected) > 1 else 0
    hr_true = 60000 / rr_true
    hr_det = 60000 / rr_det if rr_det > 0 else 0
    
    print(f"Total True Beats: {len(ground_truth)}")
    print(f"Total Detected:   {len(detected)}")
    print(f"True Positives:   {tp}")
    print(f"False Positives:  {fp}")
    print(f"False Negatives:  {fn}")
    print(f"Sensitivity:      {sensitivity*100:.2f}%")
    print(f"Positive Pred (PPV): {ppv*100:.2f}%")
    print(f"Heart Rate Error: {abs(hr_true - hr_det):.2f} BPM\n")


def print_ml_metrics():
    print("=========================================================================")
    print("   END-TO-END ML CLASSIFICATION (5-FOLD STRATIFIED GROUP CV)             ")
    print("=========================================================================\n")
    
    res = run_full_ml_benchmark(n_samples_per_class=12)
    ds = res['dataset_info']
    raw = res['raw_model_metrics']
    filt = res['filtered_model_metrics']
    
    print("--- DATASET AUDIT ---")
    print(f"Unique ECG Segments (Groups): {ds['n_segments']}")
    print(f"Extracted Beats (Raw): {ds['n_raw_beats']}")
    print(f"Extracted Beats (Filtered): {ds['n_filt_beats']}")
    print("Class Distribution (Filtered):")
    for name, count in zip(CLASS_NAMES, ds['y_filt_counts']):
        pct = (count / ds['n_filt_beats']) * 100 if ds['n_filt_beats'] > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")
    print("\n")
    
    print("--- 5-FOLD CROSS VALIDATION RESULTS ---")
    df_cv = pd.DataFrame([
        {
            'Pipeline': 'Raw / Noisy',
            'Accuracy': f"{raw['accuracy']*100:.1f} ± {raw['accuracy_std']*100:.1f}%",
            'Precision': f"{raw['precision']*100:.1f} ± {raw['precision_std']*100:.1f}%",
            'Recall': f"{raw['recall']*100:.1f} ± {raw['recall_std']*100:.1f}%",
            'F1-Score': f"{raw['f1_score']*100:.1f} ± {raw['f1_score_std']*100:.1f}%",
        },
        {
            'Pipeline': 'DSP Filtered',
            'Accuracy': f"{filt['accuracy']*100:.1f} ± {filt['accuracy_std']*100:.1f}%",
            'Precision': f"{filt['precision']*100:.1f} ± {filt['precision_std']*100:.1f}%",
            'Recall': f"{filt['recall']*100:.1f} ± {filt['recall_std']*100:.1f}%",
            'F1-Score': f"{filt['f1_score']*100:.1f} ± {filt['f1_score_std']*100:.1f}%",
        }
    ])
    print(df_cv.to_string(index=False))
    print("\n")
    
    acc_gain = (filt['accuracy'] - raw['accuracy']) * 100
    print(f">> Measured Accuracy Gain from DSP Pipeline: +{acc_gain:.2f}%\n")
    
    # Save Confusion Matrices
    np.save('cm_raw.npy', res['cm_raw'])
    np.save('cm_filt.npy', res['cm_filt'])
    print("Saved confusion matrices to cm_raw.npy and cm_filt.npy")
    
    with open('ecg_model.pkl', 'wb') as f:
        pickle.dump(res['trained_filtered_model'], f)
    print("Saved final trained model to ecg_model.pkl\n")


def main():
    evaluate_dsp_noise_levels()
    evaluate_pan_tompkins()
    print_ml_metrics()
    print("Phase 2 Rigorous Validation Completed Successfully!")


if __name__ == '__main__':
    main()
