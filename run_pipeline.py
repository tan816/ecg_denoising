"""
Full Pipeline Execution Script (Stage 5 Verification)
Runs end-to-end signal processing, QRS detection, feature extraction, ML training,
and prints comparative raw vs filtered performance benchmarks.
"""

import pickle
import numpy as np
import pandas as pd
from ml_classifier import run_full_ml_benchmark, CLASS_NAMES


def main():
    print("=========================================================================")
    print("   END-TO-END ECG SIGNAL PROCESSING & ML DIAGNOSIS PIPELINE BENCHMARK    ")
    print("=========================================================================\n")

    benchmark_res = run_full_ml_benchmark(n_samples_per_class=12)

    dsp = benchmark_res['dsp_metrics']
    raw_m = benchmark_res['raw_model_metrics']
    filt_m = benchmark_res['filtered_model_metrics']

    # 1. Signal Quality Improvement Table
    print("-------------------------------------------------------------------------")
    print(" 1. DSP SIGNAL QUALITY ENHANCEMENT METRICS")
    print("-------------------------------------------------------------------------")
    df_dsp = pd.DataFrame([{
        'Input SNR (dB)': f"{dsp['avg_snr_raw']:.2f}",
        'Output SNR (dB)': f"{dsp['avg_snr_filtered']:.2f}",
        'SNR Improvement (dB)': f"+{dsp['snr_improvement_db']:.2f}",
        'Mean Squared Error (MSE)': f"{dsp['avg_mse']:.5f}",
        'Peak SNR (PSNR dB)': f"{dsp['avg_psnr']:.2f}"
    }])
    print(df_dsp.to_string(index=False))
    print("\n")

    # 2. Diagnostic Accuracy Comparison Table
    print("-------------------------------------------------------------------------")
    print(" 2. DIAGNOSTIC CLASSIFICATION COMPARISON (RAW vs FILTERED)")
    print("-------------------------------------------------------------------------")
    df_clf = pd.DataFrame([
        {
            'Preprocessing': 'Raw Noisy Signal',
            'Accuracy': f"{raw_m['accuracy'] * 100:.2f}%",
            'Precision': f"{raw_m['precision'] * 100:.2f}%",
            'Recall': f"{raw_m['recall'] * 100:.2f}%",
            'Macro F1-Score': f"{raw_m['f1_score'] * 100:.2f}%"
        },
        {
            'Preprocessing': 'DSP Filtered Signal (Proposed)',
            'Accuracy': f"{filt_m['accuracy'] * 100:.2f}%",
            'Precision': f"{filt_m['precision'] * 100:.2f}%",
            'Recall': f"{filt_m['recall'] * 100:.2f}%",
            'Macro F1-Score': f"{filt_m['f1_score'] * 100:.2f}%"
        }
    ])
    print(df_clf.to_string(index=False))
    print("-------------------------------------------------------------------------")
    
    accuracy_boost = (filt_m['accuracy'] - raw_m['accuracy']) * 100
    print(f">> Measurable Accuracy Gain after DSP Preprocessing: +{accuracy_boost:.2f}%\n")

    # Save trained model to disk for Streamlit app
    with open('ecg_model.pkl', 'wb') as f:
        pickle.dump(benchmark_res['trained_filtered_model'], f)
    print("Saved trained model to file: ecg_model.pkl")
    print("Pipeline Execution Completed Successfully!")


if __name__ == '__main__':
    main()
