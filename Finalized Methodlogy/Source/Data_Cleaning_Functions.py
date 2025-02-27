import pandas as pd

def concat_data():

    # Fetch historical data from Github
    hist_url = 'https://raw.githubusercontent.com/ijyliu/mcbroken-daily-historical/refs/heads/main/mcbroken_daily_most_recent_on_20250215.csv'
    historical_data = pd.read_csv(hist_url)
    historical_data = historical_data[historical_data['date'] != '2025-02-15']
    
    # Fetch recent data from S3 URL (assuming public access)
    s3_url = 'https://mcbroken-bucket.s3.us-west-1.amazonaws.com/updated-mcbroken.csv'
    recent_data = pd.read_csv(s3_url)
    
    # Stack data
    lim_hist = historical_data[['date', 'broken_machines', 'total_machines']]
    lim_recent = recent_data[['date', 'broken_machines', 'total_machines']]
    df = pd.concat([lim_hist, lim_recent], axis=0).sort_values('date').reset_index(drop=True)

    return df

def clean_data(df):

    # Rename columns
    df = df.rename(columns={
        'broken_machines': 'Broken Machines',
        'total_machines': 'Total Machines',
        'date': 'Date'
    })
    df['Percent Broken'] = df['Broken Machines'] / df['Total Machines'] * 100
    df['Revenue Losses'] = df['Broken Machines'] * 625
    
    # Add missing days
    df['Date'] = pd.to_datetime(df['Date'])
    all_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    all_dates_df = pd.DataFrame(all_dates, columns=['Date'])
    df = pd.merge(all_dates_df, df, on='Date', how='left').sort_values('Date').reset_index(drop=True)
    
    # Outlier detection using 7-day moving average approach
    # Calculate 7-day moving average of Total Machines
    df['7DMA'] = df['Total Machines'].rolling(window=7).mean()
    
    # Flag initial outliers where Total Machines drops below 80% of 7DMA
    df['Outlier'] = df['Total Machines'] < df['7DMA'] * 0.8
    
    # Create column to store 7DMA values only for outlier periods
    df['Outlier 7DMA'] = df['7DMA']
    df.loc[~df['Outlier'], 'Outlier 7DMA'] = None
    # Forward fill to maintain the pre-outlier 7DMA threshold
    df['Outlier 7DMA'] = df['Outlier 7DMA'].fillna(method='ffill')
    
    # Extend outlier periods until Total Machines recovers above the pre-outlier level
    # This handles cases where machines come back online gradually
    flag = False  # Tracks if we're in an outlier period
    for i, row in df.iterrows():
        if row['Outlier']:  # Start of outlier period
            flag = True
        if flag:  # Continue marking as outlier while in period
            df.at[i, 'Outlier'] = True
        if row['Total Machines'] > row['Outlier 7DMA']:  # End period when machines recover
            flag = False
            
    # Manual outlier additions for known data quality issues
    # Extended outage in late 2024
    df.loc[(df['Date'] >= '2024-08-12') & (df['Date'] <= '2024-09-09'), 'Outlier'] = True
    # Single day anomalies
    df.loc[df['Date'] == '2022-09-09', 'Outlier'] = True
    df.loc[df['Date'] == '2021-10-06', 'Outlier'] = True

    # Mark any zeroes as outliers
    df.loc[df['Broken Machines'] == 0, 'Outlier'] = True

    return df
