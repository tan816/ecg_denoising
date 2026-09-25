import os
import numpy as np
import pandas as pd
import wfdb
import warnings
from dsp_filters import apply_dsp_pipeline
from qrs_detector import pan_tompkins_qrs, extract_beat_features, calculate_advanced_hrv_metrics, calculate_hrv_metrics
from data_loader import load_mitbih_record

warnings.filterwarnings("ignore")

# Standard AAMI beat annotation symbols
VALID_BEAT_SYMBOLS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'}

def match_peaks(detected_peaks, true_peaks, tolerance=54):
    """
    Match detected peaks to true peaks within a tolerance (default 54 samples = 150ms at 360Hz).
    """
    matched = 0
    errors = []
    
    # Simple greedy matching
    det_copy = list(detected_peaks)
    
    for tp in true_peaks:
        if not det_copy:
            break
        # find closest
        idx = np.argmin(np.abs(np.array(det_copy) - tp))
        dist = np.abs(det_copy[idx] - tp)
        
        if dist <= tolerance:
            matched += 1
            errors.append(dist)
            det_copy.pop(idx)
            
    TP = matched
    FN = len(true_peaks) - TP
    FP = len(detected_peaks) - TP
    
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * (sens * ppv) / (sens + ppv) if (sens + ppv) > 0 else 0.0
    mae = np.mean(errors) if errors else np.nan
    
    return TP, FP, FN, sens, ppv, f1, mae

def validate_mitbih():
    records = ['100', '101', '103', '105', '106', '108', '109', '111', '113', '117', '118', '119']
    
    # 5 minutes = 300s = 108000 samples at 360Hz. 
    # Let's use 108000 to get solid HRV stats
    SAMP_TO = 108000
    
    results = []
    
    print(f"Starting MIT-BIH Validation on {len(records)} records (up to {SAMP_TO} samples each)...")
    
    for rec_id in records:
        print(f"Processing Record {rec_id}...")
        try:
            rec_data = load_mitbih_record(rec_id, sampto=SAMP_TO)
            sig = rec_data['signal']
            fs = rec_data['fs']
            
            # Extract valid true beats
            ann_samples = rec_data['ann_sample']
            ann_symbols = rec_data['ann_symbol']
            
            true_beats = []
            for samp, sym in zip(ann_samples, ann_symbols):
                if sym in VALID_BEAT_SYMBOLS:
                    true_beats.append(samp)
            true_beats = np.array(true_beats)
            
            # 3. Validate signal processing
            nan_count = np.sum(np.isnan(sig))
            inf_count = np.sum(np.isinf(sig))
            
            # Run DSP
            dsp_res = apply_dsp_pipeline(sig, fs=fs)
            filtered_sig = dsp_res['final_filtered']
            
            len_match = len(sig) == len(filtered_sig)
            filter_success = not np.any(np.isnan(filtered_sig)) and len_match
            
            # 4. QRS Validation - RAW
            pt_raw = pan_tompkins_qrs(sig, fs=fs)
            raw_peaks = pt_raw['r_peaks']
            raw_tp, raw_fp, raw_fn, raw_sens, raw_ppv, raw_f1, raw_mae = match_peaks(raw_peaks, true_beats)
            
            # 4. QRS Validation - FILTERED
            pt_filt = pan_tompkins_qrs(filtered_sig, fs=fs)
            filt_peaks = pt_filt['r_peaks']
            filt_tp, filt_fp, filt_fn, filt_sens, filt_ppv, filt_f1, filt_mae = match_peaks(filt_peaks, true_beats)
            
            # 5. Extract Features & HRV
            if len(filt_peaks) >= 3:
                feats, clin_recs = extract_beat_features(filtered_sig, filt_peaks, fs=fs)
                rr_ms = np.diff(filt_peaks) / fs * 1000.0
                hrv_base = calculate_hrv_metrics(rr_ms)
                hrv_adv = calculate_advanced_hrv_metrics(filt_peaks, fs=fs)
                
                mean_rr = float(np.mean(rr_ms))
                heart_rate = 60000.0 / mean_rr if mean_rr > 0 else np.nan
                
                # Sanity checks
                invalid_rrs = np.sum(rr_ms <= 0)
                impossible_hr = heart_rate <= 0 or heart_rate > 300
            else:
                mean_rr = np.nan
                heart_rate = np.nan
                hrv_base = {'sdnn': np.nan, 'rmssd': np.nan, 'pnn50': np.nan}
                hrv_adv = {'lf': np.nan, 'hf': np.nan, 'lf_hf_ratio': np.nan, 'sd1': np.nan, 'sd2': np.nan}
                invalid_rrs = 0
                impossible_hr = True
                
            res_dict = {
                'Record': rec_id,
                'FS': fs,
                'Samples': len(sig),
                'NaNs': nan_count,
                'Infs': inf_count,
                'Filter_Success': filter_success,
                'True_Beats': len(true_beats),
                
                'Raw_Sens': raw_sens,
                'Raw_PPV': raw_ppv,
                'Raw_F1': raw_f1,
                
                'Filt_Sens': filt_sens,
                'Filt_PPV': filt_ppv,
                'Filt_F1': filt_f1,
                
                'Filt_Det_Beats': len(filt_peaks),
                'Valid_RRs': len(filt_peaks) - 1 if len(filt_peaks) > 0 else 0,
                'Mean_RR': mean_rr,
                'HR': heart_rate,
                'SDNN': hrv_base['sdnn'],
                'RMSSD': hrv_base['rmssd'],
                'pNN50': hrv_base['pnn50'],
                'LF': hrv_adv['lf'],
                'HF': hrv_adv['hf'],
                'LF_HF': hrv_adv['lf_hf_ratio'],
                'SD1': hrv_adv['sd1'],
                'SD2': hrv_adv['sd2'],
                
                'Invalid_RRs': invalid_rrs,
                'Impossible_HR': impossible_hr
            }
            results.append(res_dict)
            
        except Exception as e:
            print(f"Error processing record {rec_id}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv('mitbih_validation_results.csv', index=False)
    print("Saved mitbih_validation_results.csv")
    
    return df

if __name__ == '__main__':
    validate_mitbih()
