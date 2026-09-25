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
from plotly.subplots import make_subplots

from data_loader import load_mitbih_record, generate_synthetic_ecg, inject_noise, compute_snr, compute_mse, compute_psnr, HAS_WFDB
from dsp_filters import apply_dsp_pipeline, highpass_filter, notch_filter, lowpass_filter, wavelet_denoise
from qrs_detector import pan_tompkins_qrs, extract_beat_features, calculate_hrv_metrics
from ml_classifier import ECGHeartbeatClassifier, CLASS_NAMES, CLASS_MAP

# Page config
st.set_page_config(
    page_title="ECG DSP & ML Diagnosis Dashboard",
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
    # Fallback fit
    from ml_classifier import run_full_ml_benchmark
    res = run_full_ml_benchmark(n_samples_per_class=8)
    return res['trained_filtered_model']


def main():
    st.title("🫀 ECG Signal Processing & Diagnostic Machine Learning Dashboard")
    st.markdown("""
    **End-to-End Pipeline**: Biomedical DSP Noise Removal (Butterworth & DWT Wavelet) $\\rightarrow$ Pan-Tompkins QRS Peak Detection $\\rightarrow$ Clinical Feature Extraction $\\rightarrow$ 5-Class ML Heart Condition Diagnosis.
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
        pred_condition = "N/A"
        confidence = 0.0
        avg_proba = np.zeros(5)
        clin_recs = []

    # Dashboard Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Input SNR", f"{snr_in:.2f} dB")
    m2.metric("Output SNR", f"{snr_out:.2f} dB", f"+{delta_snr:.2f} dB")
    m3.metric("MSE", f"{mse_val:.4f}")
    m4.metric("PSNR", f"{psnr_val:.2f} dB")
    m5.metric("Heart Rate", f"{avg_hr:.1f} BPM")
    m6.metric("Diagnosis", f"{pred_condition}", f"{confidence:.1f}% conf")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Interactive Waveforms & QRS Peaks", 
        "🔍 Filter Stage Cascade Breakdown", 
        "📊 Frequency Spectrum (FFT)", 
        "📋 Clinical Features & ML Performance"
    ])

    time_arr = np.arange(len(raw_noisy)) / fs

    with tab1:
        st.subheader("Raw Noisy Signal vs DSP Filtered Waveform & Pan-Tompkins R-Peaks")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Raw / Noisy ECG Input Signal", "DSP Filtered ECG Signal & Detected R-Peaks"))

        # Raw plot
        fig.add_trace(go.Scatter(x=time_arr, y=raw_noisy, name="Raw Noisy Signal", line=dict(color='red', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_arr, y=clean_ecg, name="Clean Ground Truth", line=dict(color='lightgray', width=1, dash='dash')), row=1, col=1)

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
        st.subheader("Stage-by-Stage DSP Filtering Breakdown")
        dsp_cascade = apply_dsp_pipeline(raw_noisy, fs=fs)
        
        fig_cas = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                subplot_titles=(
                                    "1. High-pass Filtered (0.5 Hz) -> Baseline Wander Removed",
                                    "2. Notch Filtered (50 Hz) -> Powerline Suppressed",
                                    "3. Low-pass Filtered (45 Hz) -> High-Freq EMG Removed",
                                    "4. DWT Wavelet Denoised (db6) -> Clean Final ECG"
                                ))
        fig_cas.add_trace(go.Scatter(x=time_arr, y=dsp_cascade['hp_filtered'], line=dict(color='orange')), row=1, col=1)
        fig_cas.add_trace(go.Scatter(x=time_arr, y=dsp_cascade['notch_filtered'], line=dict(color='blue')), row=2, col=1)
        fig_cas.add_trace(go.Scatter(x=time_arr, y=dsp_cascade['lp_filtered'], line=dict(color='purple')), row=3, col=1)
        fig_cas.add_trace(go.Scatter(x=time_arr, y=dsp_cascade['final_filtered'], line=dict(color='green')), row=4, col=1)
        
        fig_cas.update_layout(height=650, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cas, use_container_width=True)

    with tab3:
        st.subheader("Frequency Spectrum Analysis (Fast Fourier Transform)")
        # FFT computation
        N_fft = len(raw_noisy)
        freqs = np.fft.rfftfreq(N_fft, 1.0/fs)
        fft_raw = np.abs(np.fft.rfft(raw_noisy))
        fft_filt = np.abs(np.fft.rfft(filtered_ecg))

        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=freqs, y=fft_raw, name="Raw Signal Spectrum", line=dict(color='red', width=1)))
        fig_fft.add_trace(go.Scatter(x=freqs, y=fft_filt, name="Filtered Signal Spectrum", line=dict(color='green', width=1.5)))
        fig_fft.update_layout(
            title="Magnitude Spectrum (0 to 100 Hz)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            xaxis=dict(range=[0, 100]),
            height=400
        )
        st.plotly_chart(fig_fft, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Extracted Heartbeat Clinical Parameters")
            st.json({
                "Detected R-Peaks Count": len(r_peaks),
                "Average Heart Rate (BPM)": round(avg_hr, 1),
                "SDNN (ms)": round(hrv['sdnn'], 2),
                "RMSSD (ms)": round(hrv['rmssd'], 2),
                "pNN50 (%)": round(hrv['pnn50'], 2),
                "Mean QRS Duration (ms)": round(float(np.mean([r['qrs_width_ms'] for r in clin_recs])) if len(clin_recs)>0 else 80.0, 1),
                "Mean PR Interval (ms)": round(float(np.mean([r['pr_interval_ms'] for r in clin_recs])) if len(clin_recs)>0 else 140.0, 1),
                "Mean QT Interval (ms)": round(float(np.mean([r['qt_interval_ms'] for r in clin_recs])) if len(clin_recs)>0 else 380.0, 1)
            })

        with c2:
            st.subheader("Machine Learning Condition Prediction")
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


if __name__ == '__main__':
    main()
