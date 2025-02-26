import pandas as pd
import boto3
import io

def lambda_handler(event, context):
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
    
    # Outlier detection
    df['7DMA'] = df['Total Machines'].rolling(window=7).mean()
    df['Outlier'] = df['Total Machines'] < df['7DMA'] * 0.8
    df['Outlier 7DMA'] = df['7DMA']
    df.loc[~df['Outlier'], 'Outlier 7DMA'] = None
    df['Outlier 7DMA'] = df['Outlier 7DMA'].fillna(method='ffill')
    flag = False
    for i, row in df.iterrows():
        if row['Outlier']:
            flag = True
        if flag:
            df.at[i, 'Outlier'] = True
        if row['Total Machines'] > row['Outlier 7DMA']:
            flag = False
    df.loc[(df['Date'] >= '2024-08-12') & (df['Date'] <= '2024-09-09'), 'Outlier'] = True
    df.loc[df['Date'] == '2022-09-09', 'Outlier'] = True
    df.loc[df['Date'] == '2021-10-06', 'Outlier'] = True
    
    # Save to S3 as Excel file
    bucket = 'mcbroken-bucket'
    key = 'Clean_McBroken_Daily.xlsx'
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer) as writer:
        df.to_excel(writer, index=False)
    excel_buffer.seek(0)
    
    s3_client = boto3.client('s3')
    s3_client.put_object(Bucket=bucket, Key=key, Body=excel_buffer.getvalue())
    
    return {
        'statusCode': 200,
        'body': 'Excel file successfully saved to S3'
    }
