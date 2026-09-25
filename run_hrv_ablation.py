import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from ml_classifier import create_cardiac_dataset, ECGHeartbeatClassifier

def evaluate_cv(X, y, groups):
    if len(X) == 0:
        return None
    sgkf = StratifiedGroupKFold(n_splits=5)
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
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
        
    return {
        'Accuracy': f"{np.mean(acc_list)*100:.2f} ± {np.std(acc_list)*100:.2f}%",
        'Precision': f"{np.mean(prec_list)*100:.2f} ± {np.std(prec_list)*100:.2f}%",
        'Recall': f"{np.mean(rec_list)*100:.2f} ± {np.std(rec_list)*100:.2f}%",
        'Macro F1': f"{np.mean(f1_list)*100:.2f} ± {np.std(f1_list)*100:.2f}%"
    }

def main():
    print("Generating Dataset...")
    X_raw, y_raw, groups_raw, X_filt, y_filt, groups_filt, _ = create_cardiac_dataset(n_samples_per_class=15)
    
    # Feature configurations
    # A: Existing [0:12]
    # B: Existing + LF/HF [0:15]
    # C: Existing + Poincare [0:12] + [15:17]
    # D: Existing + LF/HF + Poincare [0:17]
    
    configs = {
        'A (Existing)': lambda X: X[:, :12],
        'B (Existing + LF/HF)': lambda X: X[:, :15],
        'C (Existing + Poincare)': lambda X: np.hstack([X[:, :12], X[:, 15:17]]),
        'D (Existing + LF/HF + Poincare)': lambda X: X[:, :17]
    }
    
    print("\n--- HRV FEATURE ABLATION STUDY (Raw Noisy Signals) ---")
    res_raw = []
    for name, selector in configs.items():
        X_slice = selector(X_raw)
        res = evaluate_cv(X_slice, y_raw, groups_raw)
        res['Configuration'] = name
        res_raw.append(res)
    print(pd.DataFrame(res_raw).to_string(index=False))
    
    print("\n--- HRV FEATURE ABLATION STUDY (DSP Filtered Signals) ---")
    res_filt = []
    for name, selector in configs.items():
        X_slice = selector(X_filt)
        res = evaluate_cv(X_slice, y_filt, groups_filt)
        res['Configuration'] = name
        res_filt.append(res)
    print(pd.DataFrame(res_filt).to_string(index=False))

if __name__ == '__main__':
    main()
