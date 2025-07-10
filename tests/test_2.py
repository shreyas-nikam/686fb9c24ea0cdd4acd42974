import pytest
from definition_7fbf7abe48d14c3b9193dc4d27dfd766 import plot_time_series
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import MagicMock

@pytest.fixture
def sample_series():
    # Create a sample pandas Series for testing
    data = [1, 3, 5, 2, 4]
    index = pd.date_range('2023-01-01', periods=5, freq='D')
    return pd.Series(data, index=index)

def test_plot_time_series_returns_plotly_figure(sample_series):
    """Test that the function returns a Plotly Figure object."""
    figure = plot_time_series(sample_series)
    assert isinstance(figure, go.Figure)

def test_plot_time_series_correct_data(sample_series):
    """Test that the plot contains the correct data from the series."""
    figure = plot_time_series(sample_series)
    
    # Extract data from the figure and compare to the original series
    trace = figure.data[0]
    assert list(trace.y) == sample_series.tolist()
    assert list(trace.x) == sample_series.index.tolist()

def test_plot_time_series_empty_series():
    """Test that it handles an empty series without errors."""
    empty_series = pd.Series([], index=pd.to_datetime([]))
    figure = plot_time_series(empty_series)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 0
    assert len(figure.data[0].y) == 0

def test_plot_time_series_non_timeseries_index():
    """Test that function correctly handles a series with non-datetime index"""
    data = [1, 3, 5, 2, 4]
    index = [1,2,3,4,5]
    series = pd.Series(data, index=index)

    figure = plot_time_series(series)
    assert isinstance(figure, go.Figure)
    trace = figure.data[0]
    assert list(trace.y) == series.tolist()
    assert list(trace.x) == series.index.tolist()


def test_plot_time_series_sets_labels(sample_series, monkeypatch):
    """Test that function sets x and y axis labels"""
    mock_go_Figure = MagicMock()
    monkeypatch.setattr("plotly.graph_objects.Figure", mock_go_Figure)
    plot_time_series(sample_series)
    mock_go_Figure.return_value.update_layout.assert_called_once()
    args, kwargs = mock_go_Figure.return_value.update_layout.call_args
    assert "xaxis_title" in kwargs
    assert "yaxis_title" in kwargs
    
