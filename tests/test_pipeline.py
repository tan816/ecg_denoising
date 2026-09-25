import pytest
from ml_classifier import run_full_ml_benchmark

def test_pipeline():
    # Run the full pipeline with a very small number of samples
    # to ensure it executes without crashing end-to-end
    result = run_full_ml_benchmark(n_samples_per_class=1)
    
    assert 'dsp_metrics' in result
    assert 'raw_model_metrics' in result
    assert 'filtered_model_metrics' in result
    assert 'trained_filtered_model' in result
