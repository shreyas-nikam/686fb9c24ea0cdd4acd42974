import pytest
from definition_8630d74f1bb74e0982b78f21695086f7 import calculate_acf
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    # Create a sample pandas Series for testing
    np.random.seed(42)  # for reproducibility
    data = pd.Series(np.random.randn(100))
    return data

def test_calculate_acf_basic(sample_data):
    """Test with valid data and lags."""
    acf_values, conf_int = calculate_acf(sample_data, lags=10)
    assert isinstance(acf_values, np.ndarray)
    assert isinstance(conf_int, np.ndarray)
    assert len(acf_values) == 11  # Includes lag 0
    assert conf_int.shape == (11, 2)

def test_calculate_acf_empty_data():
    """Test with empty data series."""
    empty_data = pd.Series([])
    with pytest.raises(Exception):  # Expecting an exception due to empty data
        calculate_acf(empty_data, lags=5)

def test_calculate_acf_zero_lags(sample_data):
    """Test with zero lags."""
    acf_values, conf_int = calculate_acf(sample_data, lags=0)
    assert isinstance(acf_values, np.ndarray)
    assert isinstance(conf_int, np.ndarray)
    assert len(acf_values) == 1 # Includes lag 0
    assert conf_int.shape == (1, 2)
    assert acf_values[0] == 1.0 #ACF at lag 0 should always be 1

def test_calculate_acf_invalid_lags(sample_data):
    """Test with negative lags (should raise an error)."""
    with pytest.raises(ValueError):
        calculate_acf(sample_data, lags=-5)

def test_calculate_acf_non_series_input():
    """Test with non-pandas Series input."""
    with pytest.raises(TypeError):
        calculate_acf([1, 2, 3], lags=3)