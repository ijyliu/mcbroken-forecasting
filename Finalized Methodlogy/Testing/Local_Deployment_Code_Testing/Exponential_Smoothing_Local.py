import sys
import os

# Add the Source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Source'))

# Import functions
from Exponential_Smoothing_Forecast_Functions import *
from utils import load_data_local, save_output_local

# Run the forecast
if __name__ == "__main__":
    print("Loading data...")
    df = load_data_local()
    
    print("Generating Exponential Smoothing forecast...")

    # Generate forecast
    fit_hw, forecast_df, orig_df = exponential_smoothing_forecast(df)

    # Plot forecast
    fig = plot_forecast(forecast_df, orig_df)
    
    print("Saving output...")
    save_output_local(fig, "Daily_Exponential_Smoothing_Forecast.html")
    
    print("Exponential Smoothing forecast complete!")
