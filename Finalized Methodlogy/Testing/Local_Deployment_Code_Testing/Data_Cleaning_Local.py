import sys
import os

# Add the Source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Source'))

# Import functions
from Data_Cleaning_Functions import *

def main():
    print("Starting data cleaning process...")
    
    # Fetch and concatenate data
    print("Fetching and concatenating data...")
    df = concat_data()
    
    # Clean the data
    print("Cleaning data...")
    df = clean_data(df)
    
    # Save cleaned data to local directory
    print("Saving cleaned data to local directory...")
    output_file = "Clean_McBroken_Daily_Local.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Data cleaning complete! File saved to: {os.path.abspath(output_file)}")
    
    return df

if __name__ == "__main__":
    main()
