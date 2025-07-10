import pytest
from definition_313f9c67b31e4df985d9127880aa39da import calculate_acf
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    # Create a simple pandas Series for testing
    index = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = pd.Series(np.random.randn(20), index=index)
    return data

def test_calculate_acf_basic(sample_data):
    acf, conf_int = calculate_acf(sample_data, lags=5)
    assert len(acf) == 6  # Includes lag 0
    assert len(conf_int) == 2
    assert len(conf_int[0]) == 6
    assert len(conf_int[1]) == 6

def test_calculate_acf_zero_lags(sample_data):
    acf, conf_int = calculate_acf(sample_data, lags=0)
    assert len(acf) == 1
    assert len(conf_int) == 2
    assert len(conf_int[0]) == 1
    assert len(conf_int[1]) == 1
    assert acf[0] == 1.0

def test_calculate_acf_large_lags(sample_data):
    acf, conf_int = calculate_acf(sample_data, lags=len(sample_data)-1)
    assert len(acf) == len(sample_data)
    assert len(conf_int) == 2
    assert len(conf_int[0]) == len(sample_data)
    assert len(conf_int[1]) == len(sample_data)
    
def test_calculate_acf_non_series():
    with pytest.raises(TypeError):
        calculate_acf([1, 2, 3], lags=2)

def test_calculate_acf_empty_series():
    empty_series = pd.Series([])
    acf, conf_int = calculate_acf(empty_series, lags=5)
    assert len(acf) == 6
    assert all(np.isnan(x) for x in acf)
    assert len(conf_int) == 2
    assert len(conf_int[0]) == 6
    assert len(conf_int[1]) == 6
    assert all(np.isnan(x) for x in conf_int[0])
    assert all(np.isnan(x) for x in conf_int[1])
