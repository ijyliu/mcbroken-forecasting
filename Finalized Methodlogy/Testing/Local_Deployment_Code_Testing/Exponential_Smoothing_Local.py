import sys
import os

# Add the Source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Source'))

# Import functions
from Exponential_Smoothing_Forecast_Functions import *
from utils import load_data_local, save_output_local
from Image_Description_Functions import *

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

    # Image description
    print("Generating image description...")

    # Generate description for the plot
    
    # API key
    api_key = get_api_key_local()  # Use local function to get API key
    print("API key loaded successfully.")

    # Get the description of the chart
    description = get_image_description(fig, api_key)
    print("Image description generated successfully.")

    # Get HTML of description
    description_html = generate_description_html(description)
    print("HTML for image description generated successfully.")

    # Save the description to a file
    out_path = save_description_html_local(description_html, "Daily_Exponential_Smoothing_Forecast_Description.html")
    print(f"Image description saved to: {out_path}")

    print("Image description generation complete!")
