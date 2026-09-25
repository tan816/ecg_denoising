"""
Machine Learning Cardiac Classifier Engine (Stage 4 & 5)
Trains Random Forest / XGBoost classifiers to diagnose 5 heart conditions:
1. Normal Sinus Rhythm (NSR)
2. Bradycardia (HR < 60 BPM)
3. Tachycardia (HR > 100 BPM)
4. Atrial Fibrillation (AFib)
5. Premature Ventricular Contraction (PVC)

Compares diagnostic accuracy and F1 scores between Raw Noisy signals and Filtered DSP signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from data_loader import generate_synthetic_ecg, inject_noise, compute_snr, compute_mse, compute_psnr
from dsp_filters import apply_dsp_pipeline
from qrs_detector import pan_tompkins_qrs, extract_beat_features

CLASS_MAP = {
    'NSR': 0,
    'Bradycardia': 1,
    'Tachycardia': 2,
    'AFib': 3,
    'PVC': 4
}

CLASS_NAMES = ['NSR', 'Bradycardia', 'Tachycardia', 'AFib', 'PVC']


def create_cardiac_dataset(n_samples_per_class: int = 15, fs: int = 360, add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates a multi-condition ECG dataset for raw noisy and DSP filtered signals.
    Returns:
        X_raw, y_raw, groups_raw, X_filtered, y_filtered, groups_filtered, quality_metrics
    """
    X_raw_list, y_raw_list, groups_raw_list = [], [], []
    X_filt_list, y_filt_list, groups_filt_list = [], [], []
    
    snr_raw_list, snr_filt_list = [], []
    mse_list, psnr_list = [], []
    
    conditions = ['NSR', 'Bradycardia', 'Tachycardia', 'AFib', 'PVC']
    
    group_id = 0
    for cond in conditions:
        label = CLASS_MAP[cond]
        for seed in range(n_samples_per_class):
            group_id += 1
            # Generate clean ECG
            ecg_data = generate_synthetic_ecg(duration_sec=12.0, fs=fs, condition=cond, random_seed=seed * 10 + 5)
            clean_sig = ecg_data['signal']
            
            if add_noise:
                # Inject noise
                noise_data = inject_noise(clean_sig, fs=fs, random_seed=seed * 10 + 5)
                noisy_sig = noise_data['noisy_signal']
            else:
                noisy_sig = clean_sig.copy()

            # Apply DSP filtering cascade
            dsp_res = apply_dsp_pipeline(noisy_sig, fs=fs, use_wavelet=True)
            filtered_sig = dsp_res['final_filtered']

            # Calculate SNR, MSE, PSNR improvement
            snr_in = compute_snr(clean_sig, noisy_sig)
            snr_out = compute_snr(clean_sig, filtered_sig)
            mse_val = compute_mse(clean_sig, filtered_sig)
            psnr_val = compute_psnr(clean_sig, filtered_sig)

            snr_raw_list.append(snr_in)
            snr_filt_list.append(snr_out)
            mse_list.append(mse_val)
            psnr_list.append(psnr_val)

            # Feature Extraction on Raw Noisy Signal
            try:
                pt_raw = pan_tompkins_qrs(noisy_sig, fs=fs)
                if len(pt_raw['r_peaks']) >= 3:
                    feats_raw, _ = extract_beat_features(noisy_sig, pt_raw['r_peaks'], fs=fs)
                    for f in feats_raw:
                        if not np.isnan(f).any():
                            X_raw_list.append(f)
                            y_raw_list.append(label)
                            groups_raw_list.append(group_id)
            except Exception:
                pass

            # Feature Extraction on Filtered DSP Signal
            try:
                pt_filt = pan_tompkins_qrs(filtered_sig, fs=fs)
                if len(pt_filt['r_peaks']) >= 3:
                    feats_filt, _ = extract_beat_features(filtered_sig, pt_filt['r_peaks'], fs=fs)
                    for f in feats_filt:
                        if not np.isnan(f).any():
                            X_filt_list.append(f)
                            y_filt_list.append(label)
                            groups_filt_list.append(group_id)
            except Exception:
                pass

    X_raw = np.array(X_raw_list)
    y_raw = np.array(y_raw_list)
    groups_raw = np.array(groups_raw_list)
    
    X_filt = np.array(X_filt_list)
    y_filt = np.array(y_filt_list)
    groups_filt = np.array(groups_filt_list)

    quality_metrics = {
        'avg_snr_raw': float(np.mean(snr_raw_list)),
        'avg_snr_filtered': float(np.mean(snr_filt_list)),
        'snr_improvement_db': float(np.mean(snr_filt_list) - np.mean(snr_raw_list)),
        'avg_mse': float(np.mean(mse_list)),
        'avg_psnr': float(np.mean(psnr_list))
    }

    return X_raw, y_raw, groups_raw, X_filt, y_filt, groups_filt, quality_metrics


class ECGHeartbeatClassifier:
    """
    Random Forest / XGBoost Classifier wrapper for ECG heartbeat condition diagnosis.
    """
    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        if model_type == 'xgb' and HAS_XGB:
            self.model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        if len(y_test) == 0:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'confusion_matrix': np.zeros((5, 5))
            }
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2, 3, 4])

        return {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm
        }


def run_full_ml_benchmark(n_samples_per_class: int = 15) -> Dict[str, Any]:
    """
    Runs complete dataset creation, trains classifiers on Raw vs Filtered signals using 5-Fold Stratified Group CV,
    and returns comparative diagnostic accuracy table, signal quality metrics, and confusion matrices.
    """
    print("Generating ECG dataset & extracting features...")
    X_raw, y_raw, groups_raw, X_filt, y_filt, groups_filt, dsp_metrics = create_cardiac_dataset(n_samples_per_class=n_samples_per_class)

    sgkf = StratifiedGroupKFold(n_splits=5)
    
    def evaluate_cv(X, y, groups):
        if len(X) == 0:
             return None, None
        acc_list, prec_list, rec_list, f1_list = [], [], [], []
        cm_total = np.zeros((5, 5))
        for train_idx, test_idx in sgkf.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            clf = ECGHeartbeatClassifier(model_type='rf')
            clf.train(X_tr, y_tr)
            eval_res = clf.evaluate(X_te, y_te)
            acc_list.append(eval_res['accuracy'])
            prec_list.append(eval_res['precision'])
            rec_list.append(eval_res['recall'])
            f1_list.append(eval_res['f1_score'])
            cm_total += eval_res['confusion_matrix']
        
        return {
            'accuracy': float(np.mean(acc_list)),
            'accuracy_std': float(np.std(acc_list)),
            'precision': float(np.mean(prec_list)),
            'precision_std': float(np.std(prec_list)),
            'recall': float(np.mean(rec_list)),
            'recall_std': float(np.std(rec_list)),
            'f1_score': float(np.mean(f1_list)),
            'f1_score_std': float(np.std(f1_list)),
            'confusion_matrix': cm_total
        }, cm_total

    eval_raw, cm_raw = evaluate_cv(X_raw, y_raw, groups_raw)
    eval_filt, cm_filt = evaluate_cv(X_filt, y_filt, groups_filt)

    # Train a final model on 70% data for the Streamlit dashboard
    all_groups = np.unique(groups_filt)
    np.random.seed(42)
    np.random.shuffle(all_groups)
    split_idx = int(len(all_groups) * 0.7)
    train_groups = set(all_groups[:split_idx])
    train_idx_filt = np.array([i for i, g in enumerate(groups_filt) if g in train_groups])
    clf_final = ECGHeartbeatClassifier(model_type='rf')
    if len(train_idx_filt) > 0:
        clf_final.train(X_filt[train_idx_filt], y_filt[train_idx_filt])

    return {
        'dsp_metrics': dsp_metrics,
        'raw_model_metrics': eval_raw,
        'filtered_model_metrics': eval_filt,
        'trained_filtered_model': clf_final,
        'cm_raw': cm_raw,
        'cm_filt': cm_filt,
        'dataset_info': {
             'n_segments': len(np.unique(groups_raw)),
             'n_raw_beats': len(X_raw),
             'n_filt_beats': len(X_filt),
             'y_raw_counts': np.bincount(y_raw, minlength=5).tolist() if len(y_raw) > 0 else [],
             'y_filt_counts': np.bincount(y_filt, minlength=5).tolist() if len(y_filt) > 0 else []
        }
    }
