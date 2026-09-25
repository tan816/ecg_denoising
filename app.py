"""
Streamlit Interactive Dashboard (Stage 6)
Visualizes Raw vs Filtered ECG signals, intermediate DSP filtering cascades,
detected Pan-Tompkins R-peaks, clinical feature metrics, and ML cardiac condition diagnosis with confidence.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_loader import load_mitbih_record, generate_synthetic_ecg, inject_noise, compute_snr, compute_mse, compute_psnr, HAS_WFDB
from dsp_filters import apply_dsp_pipeline, highpass_filter, notch_filter, lowpass_filter, wavelet_denoise
from qrs_detector import pan_tompkins_qrs, extract_beat_features, calculate_hrv_metrics, calculate_advanced_hrv_metrics
from ml_classifier import ECGHeartbeatClassifier, CLASS_NAMES, CLASS_MAP

# Page config
st.set_page_config(
    page_title="BreakNoHeart",
    page_icon="🫀",
    layout="wide"
)

@st.cache_resource
def load_trained_classifier():
    if os.path.exists('ecg_model.pkl'):
        try:
            with open('ecg_model.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    from ml_classifier import run_full_ml_benchmark
    res = run_full_ml_benchmark(n_samples_per_class=8)
    return res['trained_filtered_model']

def plot_confusion_matrix(cm, title):
    fig = px.imshow(cm,
                    labels=dict(x="Predicted Condition", y="True Condition", color="Count"),
                    x=CLASS_NAMES,
                    y=CLASS_NAMES,
                    color_continuous_scale="Blues",
                    text_auto=True)
    fig.update_layout(title=title, width=500, height=500)
    return fig

def main():
    st.title("🫀 ECG Signal Processing & Diagnostic Machine Learning Dashboard")
    
    st.markdown("""
    **Methodology & Limitations:**
    This pipeline utilizes a synthetic dataset containing controlled mathematical rhythm patterns (e.g., deterministic variance in AFib and PVCs). The advanced HRV features (LF/HF, SD1, SD2) strongly expose these mathematically rigid patterns, which is why the classifier achieves very high (up to 100%) performance. This 100% synthetic classification accuracy validates the DSP and feature extraction math, but is **not** equivalent to real-world clinical performance where biological variance is far higher. Real MIT-BIH data can be loaded for compatibility and generalization checks.
    """)

    # Load Model
    clf = load_trained_classifier()

    # Sidebar Controls
    st.sidebar.header("⚙️ Signal Acquisition Settings")
    data_source = st.sidebar.radio("Data Source", ["Synthetic Generator", "PhysioNet MIT-BIH Record"])

    if data_source == "PhysioNet MIT-BIH Record":
        if not HAS_WFDB:
            st.sidebar.error("`wfdb` package not detected. Switch to Synthetic Generator or run `pip install wfdb`.")
            rec_id = "100"
        else:
            rec_id = st.sidebar.selectbox("Select MIT-BIH Record", ["100", "105", "106", "119", "200", "203", "207"])
        sample_dur = st.sidebar.slider("Signal Duration (seconds)", 5, 20, 10)
        
        # Load Record
        rec_data = load_mitbih_record(rec_id, sampto=sample_dur*360)
        clean_ecg = rec_data['signal']
        fs = rec_data['fs']
        target_cond = rec_data.get('condition', 'NSR')
    else:
        target_cond = st.sidebar.selectbox(
            "Cardiac Condition", 
            ["NSR", "Bradycardia", "Tachycardia", "AFib", "PVC"]
        )
        sample_dur = st.sidebar.slider("Signal Duration (seconds)", 5, 20, 10)
        fs = 360
        rec_data = generate_synthetic_ecg(duration_sec=sample_dur, fs=fs, condition=target_cond)
        clean_ecg = rec_data['signal']

    # Noise Injection Controls
    st.sidebar.header("🔊 Noise Injection & Artifact Controls")
    add_noise = st.sidebar.checkbox("Inject Noise Artifacts", value=True)
    
    if add_noise:
        bw_amp = st.sidebar.slider("Baseline Wander Amp (mV)", 0.0, 1.0, 0.40, 0.05)
        pl_amp = st.sidebar.slider("Powerline 50Hz Amp (mV)", 0.0, 0.8, 0.25, 0.05)
        emg_amp = st.sidebar.slider("EMG High-Freq Amp (mV)", 0.0, 0.5, 0.15, 0.02)
        em_amp = st.sidebar.slider("Electrode Motion Amp (mV)", 0.0, 0.8, 0.20, 0.05)

        noise_res = inject_noise(
            clean_ecg, fs=fs,
            baseline_wander_amp=bw_amp,
            powerline_amp=pl_amp,
            emg_amp=emg_amp,
            electrode_motion_amp=em_amp
        )
        raw_noisy = noise_res['noisy_signal']
    else:
        raw_noisy = clean_ecg.copy()

    # Filtering Controls
    st.sidebar.header("🧹 DSP Preprocessing Filters")
    use_hp = st.sidebar.checkbox("0.5 Hz High-pass (Baseline Wander)", value=True)
    use_notch = st.sidebar.checkbox("50 Hz Notch (Powerline)", value=True)
    use_lp = st.sidebar.checkbox("45 Hz Low-pass (EMG Noise)", value=True)
    use_dwt = st.sidebar.checkbox("Wavelet Denoising (DWT db6)", value=True)

    # Process Signal
    sig = raw_noisy.copy()
    if use_hp:
        sig = highpass_filter(sig, fs=fs, cutoff=0.5)
    if use_notch:
        sig = notch_filter(sig, fs=fs, notch_freq=50.0)
    if use_lp:
        sig = lowpass_filter(sig, fs=fs, cutoff=45.0)
    if use_dwt:
        sig = wavelet_denoise(sig, wavelet='db6', level=5)
    filtered_ecg = sig

    # Compute Signal Quality Metrics
    snr_in = compute_snr(clean_ecg, raw_noisy)
    snr_out = compute_snr(clean_ecg, filtered_ecg)
    delta_snr = snr_out - snr_in
    mse_val = compute_mse(clean_ecg, filtered_ecg)
    psnr_val = compute_psnr(clean_ecg, filtered_ecg)

    # QRS Detection & Feature Extraction
    pt_res = pan_tompkins_qrs(filtered_ecg, fs=fs)
    r_peaks = pt_res['r_peaks']

    if len(r_peaks) >= 3:
        feats, clin_recs = extract_beat_features(filtered_ecg, r_peaks, fs=fs)
        rr_ms = np.diff(r_peaks) / fs * 1000.0
        hrv = calculate_hrv_metrics(rr_ms)
        adv_hrv = calculate_advanced_hrv_metrics(r_peaks, fs=fs)
        avg_hr = float(np.mean([r['hr_bpm'] for r in clin_recs]))

        # Predict ML Condition
        preds = clf.predict(feats)
        probas = clf.predict_proba(feats)
        avg_proba = np.mean(probas, axis=0)
        pred_class_idx = int(np.argmax(avg_proba))
        pred_condition = CLASS_NAMES[pred_class_idx]
        confidence = float(avg_proba[pred_class_idx] * 100.0)
    else:
        avg_hr = 0.0
        hrv = {'sdnn': 0.0, 'rmssd': 0.0, 'pnn50': 0.0}
        adv_hrv = {'lf': np.nan, 'hf': np.nan, 'lf_hf_ratio': np.nan, 'sd1': np.nan, 'sd2': np.nan, 'poincare_coords': ([], [])}
        pred_condition = "N/A"
        confidence = 0.0
        avg_proba = np.zeros(5)
        clin_recs = []

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Signal Processing", 
        "🔍 DSP Analysis", 
        "📋 QRS & HRV",
        "🧠 Diagnosis",
        "📊 Model Evaluation"
    ])

    time_arr = np.arange(len(raw_noisy)) / fs

    with tab1:
        st.subheader("Signal Processing")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Raw / Noisy Signal", "Filtered Signal"))

        # Raw plot
        fig.add_trace(go.Scatter(x=time_arr, y=raw_noisy, name="Raw Noisy Signal", line=dict(color='red', width=1)), row=1, col=1)

        # Filtered plot
        fig.add_trace(go.Scatter(x=time_arr, y=filtered_ecg, name="DSP Filtered Signal", line=dict(color='green', width=1.5)), row=2, col=1)
        if len(r_peaks) > 0:
            fig.add_trace(go.Scatter(
                x=time_arr[r_peaks], y=filtered_ecg[r_peaks],
                mode='markers', name='Detected R-Peaks',
                marker=dict(color='magenta', size=9, symbol='diamond')
            ), row=2, col=1)
            
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (mV)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("DSP Analysis & Signal Quality")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Input SNR", f"{snr_in:.2f} dB")
        m2.metric("Output SNR", f"{snr_out:.2f} dB", f"{delta_snr:+.2f} dB")
        m3.metric("MSE", f"{mse_val:.4f}")
        m4.metric("PSNR", f"{psnr_val:.2f} dB")
        
        st.subheader("Frequency Spectrum Analysis (Fast Fourier Transform)")
        # FFT computation
        N_fft = len(raw_noisy)
        freqs = np.fft.rfftfreq(N_fft, 1.0/fs)
        fft_raw = np.abs(np.fft.rfft(raw_noisy))
        fft_filt = np.abs(np.fft.rfft(filtered_ecg))

        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=freqs, y=fft_raw, name="FFT before filtering", line=dict(color='red', width=1)))
        fig_fft.add_trace(go.Scatter(x=freqs, y=fft_filt, name="FFT after filtering", line=dict(color='green', width=1.5)))
        fig_fft.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            xaxis=dict(range=[0, 100]),
            height=400
        )
        st.plotly_chart(fig_fft, use_container_width=True)

    with tab3:
        st.subheader("QRS & Heart Rate Variability")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Traditional Features**")
            st.json({
                "Heart Rate (BPM)": round(avg_hr, 1),
                "Mean RR Interval (ms)": round(float(np.mean(rr_ms)) if len(r_peaks) > 1 else 0.0, 1),
                "SDNN (ms)": round(hrv['sdnn'], 2),
                "RMSSD (ms)": round(hrv['rmssd'], 2),
                "pNN50 (%)": round(hrv['pnn50'], 2),
                "Mean QRS Duration (ms)": round(float(np.mean([r['qrs_width_ms'] for r in clin_recs if not np.isnan(r['qrs_width_ms'])])) if len(clin_recs)>0 else np.nan, 1),
                "Mean PR Interval (ms)": round(float(np.mean([r['pr_interval_ms'] for r in clin_recs if not np.isnan(r['pr_interval_ms'])])) if len(clin_recs)>0 else np.nan, 1),
                "Mean QT Interval (ms)": round(float(np.mean([r['qt_interval_ms'] for r in clin_recs if not np.isnan(r['qt_interval_ms'])])) if len(clin_recs)>0 else np.nan, 1)
            })

            st.markdown("**Advanced Frequency HRV**")
            st.json({
                "LF Power": round(adv_hrv['lf'], 3) if not np.isnan(adv_hrv['lf']) else "NaN",
                "HF Power": round(adv_hrv['hf'], 3) if not np.isnan(adv_hrv['hf']) else "NaN",
                "LF/HF Ratio": round(adv_hrv['lf_hf_ratio'], 3) if not np.isnan(adv_hrv['lf_hf_ratio']) else "NaN"
            })
            
            st.markdown("**Poincaré Features**")
            st.json({
                "SD1 (Short-term variance)": round(adv_hrv['sd1'], 3) if not np.isnan(adv_hrv['sd1']) else "NaN",
                "SD2 (Long-term variance)": round(adv_hrv['sd2'], 3) if not np.isnan(adv_hrv['sd2']) else "NaN"
            })

        with c2:
            st.markdown("**Poincaré Plot**")
            rr_n, rr_n1 = adv_hrv['poincare_coords']
            if len(rr_n) > 0:
                fig_poincare = go.Figure()
                fig_poincare.add_trace(go.Scatter(
                    x=rr_n, y=rr_n1, mode='markers', name='RR intervals',
                    marker=dict(color='blue', size=8, opacity=0.7)
                ))
                # Add line of identity
                min_rr = min(min(rr_n), min(rr_n1)) * 0.95
                max_rr = max(max(rr_n), max(rr_n1)) * 1.05
                fig_poincare.add_trace(go.Scatter(
                    x=[min_rr, max_rr], y=[min_rr, max_rr], 
                    mode='lines', name='Line of Identity', 
                    line=dict(color='gray', dash='dash')
                ))
                fig_poincare.update_layout(
                    xaxis_title="RR_n (ms)",
                    yaxis_title="RR_n+1 (ms)",
                    height=450,
                    showlegend=False
                )
                st.plotly_chart(fig_poincare, use_container_width=True)
            else:
                st.info("Not enough RR intervals to plot Poincaré.")

    with tab4:
        st.subheader("Machine Learning Diagnosis")
        st.markdown(f"**Model prediction on the evaluated dataset:** `{pred_condition}`")
        
        df_prob = pd.DataFrame({
            'Cardiac Condition': CLASS_NAMES,
            'Probability': avg_proba * 100.0
        })
        fig_prob = go.Figure(go.Bar(
            x=df_prob['Probability'],
            y=df_prob['Cardiac Condition'],
            orientation='h',
            marker=dict(color=['#2ecc71' if c == pred_condition else '#3498db' for c in CLASS_NAMES])
        ))
        fig_prob.update_layout(title="Prediction Probability Distribution (%)", xaxis_title="Probability (%)", height=320)
        st.plotly_chart(fig_prob, use_container_width=True)

    with tab5:
        st.subheader("Model Evaluation (5-Fold StratifiedGroupKFold)")
        st.markdown("The table below highlights ablation study metrics, comparing model performance across configurations.")
        
        # Hardcoded results from the ablation study and pipeline tests
        eval_data = {
            "Configuration": ["Raw + 12 features", "DSP + 12 features", "Raw + 17 features", "DSP + 17 features"],
            "Accuracy": ["90.44 ± 0.51%", "93.21 ± 0.40%", "97.91 ± 4.18%", "100.00 ± 0.00%"],
            "Precision": ["90.35 ± 0.70%", "93.64 ± 0.67%", "97.98 ± 4.03%", "100.00 ± 0.00%"],
            "Recall": ["89.34 ± 0.54%", "92.45 ± 0.44%", "97.20 ± 5.59%", "100.00 ± 0.00%"],
            "Macro F1": ["89.41 ± 0.58%", "92.26 ± 0.42%", "97.35 ± 5.29%", "100.00 ± 0.00%"]
        }
        st.table(pd.DataFrame(eval_data))
        
        col1, col2 = st.columns(2)
        try:
            if os.path.exists('cm_raw.npy') and os.path.exists('cm_filt.npy'):
                cm_raw = np.load('cm_raw.npy')
                cm_filt = np.load('cm_filt.npy')
                
                with col1:
                    st.plotly_chart(plot_confusion_matrix(cm_raw, "Raw Noisy Signal (17 features)"), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_confusion_matrix(cm_filt, "DSP Filtered Signal (17 features)"), use_container_width=True)
            else:
                st.warning("Confusion matrix files (cm_raw.npy, cm_filt.npy) not found. Run the pipeline to generate them.")
        except Exception as e:
            st.error(f"Error loading confusion matrices: {e}")

if __name__ == '__main__':
    main()
