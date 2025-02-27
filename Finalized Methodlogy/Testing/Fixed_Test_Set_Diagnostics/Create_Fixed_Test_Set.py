# Import functions from Source/Data_Cleaning_Functions.py
import sys
import os
import pandas as pd

# Add the Source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Source'))
from Data_Cleaning_Functions import *

print("Starting fixed test set creation...")

# Load data 
print("Fetching and concatenating data...")
df = concat_data()

# Limit data: 2020-10-25 to 2025-02-22
print("Limiting data to desired date range...")
df = df[(df['date'] >= '2020-10-25') & (df['date'] <= '2025-02-22')]
print(f"Data shape after limiting: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Clean data
print("Cleaning data...")
df = clean_data(df)

# Create Train column
print("Creating train/test split...")
df['Train'] = True
# We will be making 30-day ahead forecasts, so last 30 days should be test
df.loc[df['Date'] > df['Date'].max() - pd.Timedelta(days=30), 'Train'] = False
# Print date ranges by Train
print("Train/Test splits:")
print(df.groupby('Train')['Date'].agg(['min', 'max']))

# Save to Excel
output_file = os.path.join(os.path.dirname(__file__), 'Fixed_Test_Set.xlsx')
print(f"Saving fixed test set to: {output_file}")
df.to_excel(output_file, index=False)
print("Fixed test set creation complete!")
