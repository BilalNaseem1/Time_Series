import numpy as np
import pandas as pd
import plotly.express as px
import timesynth as ts

def plot_time_series(time, values, label, legends=None, width=900, height=500, save_path=None):
    """
    Plot a time series using Plotly.
    
    Parameters:
        time (array-like): Time values
        values (array-like or list of array-like): Value(s) to plot
        label (str): Title of the plot
        legends (list of str, optional): Labels for each series if values is a list
        width (int): Width of the figure
        height (int): Height of the figure
        save_path (str, optional): If provided, saves the figure to this path
    
    Returns:
        plotly.graph_objs._figure.Figure: The plotly figure object
    """
    if legends is not None:
        assert isinstance(values, list), "If legends are provided, values must be a list"
        assert len(legends) == len(values), "Legends and values must be of the same length"
    
    if isinstance(values, list):
        series_dict = {"Time": time}
        for v, l in zip(values, legends):
            series_dict[l] = v
        plot_df = pd.DataFrame(series_dict)
        plot_df = pd.melt(plot_df, id_vars="Time", var_name="ts", value_name="Value")
        fig = px.line(plot_df, x="Time", y="Value", line_dash="ts")
    else:
        series_dict = {"Time": time, "Value": values, "ts": ""}
        plot_df = pd.DataFrame(series_dict)
        fig = px.line(plot_df, x="Time", y="Value")
    
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        title={
            'text': label,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 25}
        },
        yaxis=dict(
            title=dict(
                text="Value",
                font=dict(size=12)
            )
        ),
        xaxis=dict(
            title=dict(
                text="Time",
                font=dict(size=12)
            )
        )
    )
    
    if save_path:
        fig.write_image(save_path)  # Requires 'kaleido' package
    
    return fig

def generate_timeseries(signal, noise=None, stop_time=20, num_points=100):
    """
    Generate a synthetic time series using timesynth.
    
    Parameters:
        signal: A TimeSynth signal generator
        noise: A TimeSynth noise generator (optional)
        stop_time (float): Total time duration
        num_points (int): Number of points to sample
    
    Returns:
        samples (np.ndarray): Sampled values (signal + noise)
        regular_time_samples (np.ndarray): Sampled time values
        signals (np.ndarray): Pure signal values
        errors (np.ndarray): Noise values
    """
    time_sampler = ts.TimeSampler(stop_time=stop_time)
    regular_time_samples = time_sampler.sample_regular_time(num_points=num_points)
    timeseries = ts.TimeSeries(signal_generator=signal, noise_generator=noise)
    samples, signals, errors = timeseries.sample(regular_time_samples)
    return samples, regular_time_samples, signals, errors

# Example usage:
if __name__ == "__main__":
    # Generate the time axis with sequential numbers up to 200
    time = np.arange(200)
    
    # Sample 200 random values
    values = np.random.randn(200) * 100
    
    # Create and display the plot
    fig = plot_time_series(time, values, "Random Time Series Example")
    fig.show()
    
    # Example with timesynth - generating a sinusoidal signal with noise
    signal = ts.signals.Sinusoidal(frequency=0.25)
    noise = ts.noise.GaussianNoise(std=0.3)
    
    samples, time_samples, signals, errors = generate_timeseries(
        signal=signal, 
        noise=noise, 
        stop_time=20, 
        num_points=100
    )
    
    # Plot the synthetic time series
    fig2 = plot_time_series(
        time_samples, 
        [signals, samples], 
        "Synthetic Time Series: Signal vs Signal + Noise",
        legends=["Pure Signal", "Signal + Noise"]
    )
    fig2.show()