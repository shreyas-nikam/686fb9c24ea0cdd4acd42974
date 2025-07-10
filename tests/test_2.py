import pytest
import pandas as pd
import plotly.graph_objects as go
from definition_c3f389b2906f46cb8d27fe5a7610d639 import plot_time_series

@pytest.fixture
def sample_data():
    # Create a sample pandas Series for testing
    data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2023-01-01', periods=5))
    return data

def test_plot_time_series_returns_figure(sample_data):
    """Test that the function returns a Plotly Figure object."""
    figure = plot_time_series(sample_data)
    assert isinstance(figure, go.Figure)

def test_plot_time_series_correct_data(sample_data):
    """Test that the plot contains the correct data."""
    figure = plot_time_series(sample_data)
    # Extract data from the figure and compare (handling possible variations in data structure)
    figure_data = figure.data[0].y.tolist()

    assert figure_data == sample_data.tolist()

def test_plot_time_series_handles_empty_series():
    """Test the function handles empty Pandas Series."""
    empty_series = pd.Series([])
    figure = plot_time_series(empty_series)
    assert isinstance(figure, go.Figure)

def test_plot_time_series_with_non_numeric_data():
    """Test that the function raises a TypeError if data is non-numeric."""
    non_numeric_data = pd.Series(['a', 'b', 'c'])
    with pytest.raises(TypeError):
        plot_time_series(non_numeric_data)

def test_plot_time_series_sets_axis_labels(sample_data):
    """Test that the plot sets appropriate axis labels."""
    figure = plot_time_series(sample_data)
    assert figure.layout.xaxis.title.text == "Time" or figure.layout.xaxis.title.text == "Date"
    assert figure.layout.yaxis.title.text == "MA Value"
