import pytest
from definition_cf359165f38144a981f975e3b13374ff import generate_ma_data
import pandas as pd
import numpy as np

@pytest.mark.parametrize("order, theta_coeffs, epsilon_std_dev, num_samples, expected_len", [
    (1, [0.5], 1.0, 24, 24),
    (2, [0.5, 0.2], 0.5, 48, 48),
    (1, [0.0], 2.0, 12, 12),
    (2, [0.0, 0.0], 1.0, 36, 36),
    (1, [-0.8], 0.1, 60, 60),
])
def test_generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples, expected_len):
    result = generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples)
    assert isinstance(result, pd.Series)
    assert len(result) == expected_len
    assert isinstance(result.index, pd.DatetimeIndex)

def test_generate_ma_data_invalid_order():
    with pytest.raises(ValueError):
        generate_ma_data(3, [0.5, 0.2, 0.1], 1.0, 24)

def test_generate_ma_data_incorrect_theta_length_order_1():
    with pytest.raises(ValueError):
        generate_ma_data(1, [0.5, 0.2], 1.0, 24)

def test_generate_ma_data_incorrect_theta_length_order_2():
    with pytest.raises(ValueError):
        generate_ma_data(2, [0.5], 1.0, 24)
    
def test_generate_ma_data_zero_std():
        result = generate_ma_data(1, [0.5], 0, 24)
        assert (result.values == 0).all()
