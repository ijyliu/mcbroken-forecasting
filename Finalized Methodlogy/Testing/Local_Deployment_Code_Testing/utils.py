import os
import pandas as pd

def load_data_local():
    """
    Load data from the Clean_McBroken_Daily_Local.xlsx file in the local testing directory.
    
    Returns:
        pandas.DataFrame: The loaded data
    """
    # Load data from the local testing directory
    df = pd.read_excel("Clean_McBroken_Daily_Local.xlsx")
    return df

def save_output_local(fig, filename):
    """
    Save a plotly figure to the Graphics directory with the specified filename.
    
    Args:
        fig: The plotly figure to save
        filename: The filename to save the figure as (should include .html extension)
    
    Returns:
        str: The absolute path to the saved file
    """
    # Save plot as HTML to the Graphics directory
    if not os.path.exists("Graphics"):
        os.makedirs("Graphics")
    output_path = f"Graphics/{filename}"
    fig.write_html(output_path)
    print(f"Forecast saved to: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path) 
