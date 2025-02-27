import sys
import os

# Add the Source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Source'))

# Import functions
from Prophet_Forecast_Functions import *
from utils import load_data_local, save_output_local

# Run the forecast
if __name__ == "__main__":
    print("Loading data...")
    df = load_data_local()
    
    print("Generating Prophet forecast...")

    # Generate forecast
    model, forecast_df, orig_df, fitted_values = prophet_forecast(df)

    # Plot forecast
    fig = plot_forecast(forecast_df, orig_df)
    
    print("Saving output...")
    save_output_local(fig, "Daily_Prophet_Forecast.html")
    
    print("Prophet forecast complete!")
