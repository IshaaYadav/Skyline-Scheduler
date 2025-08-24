import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_flight_volume(kpi_df: pd.DataFrame, output_path: str):
    """
    Creates and saves an interactive bar chart of flight volume by hour.

    Args:
        kpi_df: DataFrame containing hourly KPI data.
        output_path: Path to save the HTML file for the plot.
    """
    print("Generating flight volume plot...")
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=kpi_df['hour'],
        y=kpi_df['total_departures'],
        name='Total Departures',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=kpi_df['hour'],
        y=kpi_df['total_arrivals'],
        name='Total Arrivals',
        marker_color='lightsalmon'
    ))

    fig.update_layout(
        barmode='stack',
        title_text='<b>Total Flight Volume by Hour</b>',
        xaxis_title='Hour of the Day (24H Format)',
        yaxis_title='Number of Flights',
        xaxis=dict(tickmode='linear'),
        legend_title_text='Flight Type',
        template='plotly_white'
    )
    
    fig.write_html(output_path)
    print(f"Saved flight volume plot to {output_path}")


def plot_delays_vs_volume(kpi_df: pd.DataFrame, output_path: str):
    """
    Creates and saves a combo chart showing average delays vs. flight volume.

    Args:
        kpi_df: DataFrame containing hourly KPI data.
        output_path: Path to save the HTML file for the plot.
    """
    print("Generating delays vs. volume plot...")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bars for total flight volume (primary y-axis)
    fig.add_trace(
        go.Bar(
            x=kpi_df['hour'], 
            y=kpi_df['total_flights'], 
            name='Total Flights',
            marker_color='lightblue'
        ),
        secondary_y=False,
    )

    # Add lines for average delays (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=kpi_df['hour'], 
            y=kpi_df['average_departure_delay'], 
            name='Avg. Departure Delay',
            mode='lines+markers',
            line=dict(color='firebrick', width=2)
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=kpi_df['hour'], 
            y=kpi_df['average_arrival_delay'], 
            name='Avg. Arrival Delay',
            mode='lines+markers',
            line=dict(color='seagreen', width=2)
        ),
        secondary_y=True,
    )

    # Set titles and axis labels
    fig.update_layout(
        title_text='<b>Average Delays vs. Flight Volume by Hour</b>',
        xaxis_title='Hour of the Day (24H Format)',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Number of Flights", secondary_y=False)
    fig.update_yaxes(title_text="Average Delay (Minutes)", secondary_y=True)
    
    fig.write_html(output_path)
    print(f"Saved delays vs. volume plot to {output_path}")


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'outputs')
    
    # Load the kpi data from the previous step
    kpi_data_path = os.path.join(output_path, '03_hourly_kpis.csv')
    if not os.path.exists(kpi_data_path):
        print(f"Error: Input file not found at {kpi_data_path}")
        print("Please run `core/kpis.py` first.")
    else:
        kpi_df = pd.read_csv(kpi_data_path)
        
        # Ensure the output directory for plots exists
        plots_dir = os.path.join(output_path, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        # Generate and save the plots
        plot_flight_volume(kpi_df, os.path.join(plots_dir, '04_flight_volume_by_hour.html'))
        plot_delays_vs_volume(kpi_df, os.path.join(plots_dir, '04_delays_vs_volume_by_hour.html'))

