import pytest
from definition_781b750f77b446c9baedbccdb445f398 import generate_ma_data
import pandas as pd
import numpy as np

@pytest.mark.parametrize("order, theta_coeffs, epsilon_std_dev, num_samples, expected_type", [
    (1, [0.5], 1.0, 24, pd.Series),
    (2, [0.5, 0.2], 0.5, 48, pd.Series),
])
def test_generate_ma_data_output_type(order, theta_coeffs, epsilon_std_dev, num_samples, expected_type):
    result = generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples)
    assert isinstance(result, expected_type)

@pytest.mark.parametrize("order, theta_coeffs, epsilon_std_dev, num_samples", [
    (1, [0.5], 1.0, 0),
    (2, [0.5, 0.2], 0.5, -1),
])
def test_generate_ma_data_invalid_num_samples(order, theta_coeffs, epsilon_std_dev, num_samples):
    with pytest.raises(ValueError):
        generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples)

@pytest.mark.parametrize("order, theta_coeffs, epsilon_std_dev, num_samples", [
    (3, [0.5], 1.0, 24),
    (0, [0.5, 0.2], 0.5, 48),
])
def test_generate_ma_data_invalid_order(order, theta_coeffs, epsilon_std_dev, num_samples):
    with pytest.raises(ValueError):
        generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples)

@pytest.mark.parametrize("order, theta_coeffs, epsilon_std_dev, num_samples", [
    (1, [0.5, 0.2], 1.0, 24),
    (2, [0.5], 0.5, 48),
])
def test_generate_ma_data_invalid_theta_coeffs_length(order, theta_coeffs, epsilon_std_dev, num_samples):
    with pytest.raises(ValueError):
        generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples)

@pytest.mark.parametrize("order, theta_coeffs, epsilon_std_dev, num_samples", [
    (1, [0.5], 1.0, 24),
    (2, [0.5, 0.2], 0.5, 48),
])
def test_generate_ma_data_output_length(order, theta_coeffs, epsilon_std_dev, num_samples):
    result = generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples)
    assert len(result) == num_samples
