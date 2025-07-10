import pytest
from definition_4cd6d29b5d67494f80fe0188793c9ee3 import update_plots
import unittest.mock
import pandas as pd
import numpy as np


def dummy_plot(data, title):
    return None

@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.generate_ma_data", return_value=pd.Series([1,2,3]))
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.calculate_acf", return_value=([0.5, 0.2, 0.1], [0.1, 0.2, 0.3]))
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.plot_time_series", side_effect=dummy_plot)
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.plot_acf", side_effect=dummy_plot)
@unittest.mock.patch("IPython.display.display")
def test_update_plots_ma1(mock_display, mock_plot_acf, mock_plot_time_series, mock_calculate_acf, mock_generate_ma_data):
    update_plots(1, 0.5, 0.2, 1.0)
    mock_generate_ma_data.assert_called_once_with(order=1, theta_coeffs=[0.5], epsilon_std_dev=1.0, num_samples=24)
    mock_calculate_acf.assert_called_once()
    mock_plot_time_series.assert_called_once()
    mock_plot_acf.assert_called_once()
    assert mock_display.call_count == 2

@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.generate_ma_data", return_value=pd.Series([1,2,3]))
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.calculate_acf", return_value=([0.5, 0.2, 0.1], [0.1, 0.2, 0.3]))
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.plot_time_series", side_effect=dummy_plot)
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.plot_acf", side_effect=dummy_plot)
@unittest.mock.patch("IPython.display.display")
def test_update_plots_ma2(mock_display, mock_plot_acf, mock_plot_time_series, mock_calculate_acf, mock_generate_ma_data):
    update_plots(2, 0.5, 0.2, 1.0)
    mock_generate_ma_data.assert_called_once_with(order=2, theta_coeffs=[0.5, 0.2], epsilon_std_dev=1.0, num_samples=24)
    mock_calculate_acf.assert_called_once()
    mock_plot_time_series.assert_called_once()
    mock_plot_acf.assert_called_once()
    assert mock_display.call_count == 2

@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.generate_ma_data", return_value=pd.Series([1,2,3]))
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.calculate_acf", return_value=([0.5, 0.2, 0.1], [0.1, 0.2, 0.3]))
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.plot_time_series", side_effect=dummy_plot)
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.plot_acf", side_effect=dummy_plot)
@unittest.mock.patch("IPython.display.display")
def test_update_plots_zero_std(mock_display, mock_plot_acf, mock_plot_time_series, mock_calculate_acf, mock_generate_ma_data):
    update_plots(1, 0.5, 0.2, 0.0)
    mock_generate_ma_data.assert_called_once_with(order=1, theta_coeffs=[0.5], epsilon_std_dev=0.0, num_samples=24)

@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.generate_ma_data", side_effect=Exception("test"))
def test_update_plots_exception(mock_generate_ma_data):
    with pytest.raises(Exception, match="test"):
        update_plots(1, 0.5, 0.2, 1.0)

@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.generate_ma_data", return_value=pd.Series([1,2,3]))
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.calculate_acf", return_value=([0.5, 0.2, 0.1], [0.1, 0.2, 0.3]))
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.plot_time_series", side_effect=dummy_plot)
@unittest.mock.patch("definition_4cd6d29b5d67494f80fe0188793c9ee3.plot_acf", side_effect=dummy_plot)
@unittest.mock.patch("IPython.display.display")
def test_update_plots_negative_theta(mock_display, mock_plot_acf, mock_plot_time_series, mock_calculate_acf, mock_generate_ma_data):
    update_plots(2, -0.5, -0.2, 1.0)
    mock_generate_ma_data.assert_called_once_with(order=2, theta_coeffs=[-0.5, -0.2], epsilon_std_dev=1.0, num_samples=24)
    mock_calculate_acf.assert_called_once()
    mock_plot_time_series.assert_called_once()
    mock_plot_acf.assert_called_once()
    assert mock_display.call_count == 2
