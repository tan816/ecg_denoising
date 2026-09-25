import numpy as np
import pytest
from ml_classifier import create_cardiac_dataset, ECGHeartbeatClassifier

def test_classifier():
    # Use a small number of samples for fast testing
    X_raw, y_raw, groups_raw, X_filt, y_filt, groups_filt, quality_metrics = create_cardiac_dataset(n_samples_per_class=2, fs=360, add_noise=True)
    
    # Check that dimensions are aligned
    assert len(X_raw) == len(y_raw) == len(groups_raw), "Raw features and labels must have the same length"
    assert len(X_filt) == len(y_filt) == len(groups_filt), "Filtered features and labels must have the same length"
    
    # Verify no missing labels
    assert not np.any(np.isnan(y_raw)), "Missing labels in raw dataset"
    assert not np.any(np.isnan(y_filt)), "Missing labels in filtered dataset"
    
    clf = ECGHeartbeatClassifier(model_type='rf')
    
    if len(X_raw) > 0:
        clf.train(X_raw, y_raw)
        eval_metrics = clf.evaluate(X_raw, y_raw)
        assert 'accuracy' in eval_metrics, "Evaluation must return metrics"
