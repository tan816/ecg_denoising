import numpy as np
import pytest
from dsp_filters import apply_dsp_pipeline
from data_loader import generate_synthetic_ecg, inject_noise, compute_snr, compute_mse

def test_dsp_filtering():
    fs = 360
    ecg_data = generate_synthetic_ecg(duration_sec=10.0, fs=fs, condition='NSR')
    clean_sig = ecg_data['signal']
    
    noise_data = inject_noise(clean_sig, fs=fs)
    noisy_sig = noise_data['noisy_signal']
    
    dsp_res = apply_dsp_pipeline(noisy_sig, fs=fs, use_wavelet=True)
    filtered_sig = dsp_res['final_filtered']
    
    assert len(filtered_sig) == len(noisy_sig), "Output length must equal input length"
    assert not np.any(np.isnan(filtered_sig)), "Filtered signal contains NaN"
    assert not np.any(np.isinf(filtered_sig)), "Filtered signal contains Inf"
    
    snr_in = compute_snr(clean_sig, noisy_sig)
    snr_out = compute_snr(clean_sig, filtered_sig)
    
    assert snr_out > snr_in, "Filtering should improve SNR"
    
    mse_val = compute_mse(clean_sig, filtered_sig)
    assert mse_val < 0.1, f"MSE is too high: {mse_val}"
