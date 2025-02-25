import boto3
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.special import inv_boxcox
from pmdarima import auto_arima
import plotly.graph_objs as go
from io import BytesIO

def lambda_handler(event, context):
    # S3 details
    s3 = boto3.client('s3')
    bucket = 'mcbroken-bucket'  # S3 bucket name
    key = 'Clean_McBroken_Daily.xlsx'  # File name in S3

    # Load data from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_excel(BytesIO(obj['Body'].read()))

    # Sort and limit to past year
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    df = df.iloc[-365:].reset_index(drop=True)

    # Mark missing cases
    df['Missing'] = df['Revenue Losses'].isnull()

    # Box-Cox Transformation on non-outlier, non-missing cases
    orig_df = df.copy()  # Save original for plotting
    no_out_missing = df[(df['Outlier'] == False) & (df['Revenue Losses'].notnull())]
    no_out_missing['Revenue Losses'], lam = stats.boxcox(no_out_missing['Revenue Losses'])
    df.loc[no_out_missing.index, 'Revenue Losses'] = no_out_missing['Revenue Losses']

    # Set missing and outlier cases to the mean of the non-outlier data
    no_outlier_mean = df.query('Outlier == False').mean()['Revenue Losses']
    df['Revenue Losses'] = df['Revenue Losses'].fillna(no_outlier_mean)
    df['Revenue Losses'] = df['Revenue Losses'].mask(df['Outlier'], no_outlier_mean)

    df = df[['Date', 'Revenue Losses', 'Missing', 'Outlier']]

    # Add indicator columns for outliers and missing values
    outlier_cols, missing_cols = [], []
    for i in range(len(df)):
        if df.loc[i, 'Outlier'] == 1:
            col_name = 'Outlier_' + str(df['Date'][i])
            df[col_name] = [1 if j == i else 0 for j in range(len(df))]
            outlier_cols.append(col_name)
        if df.loc[i, 'Missing'] == 1:
            col_name = 'Missing_' + str(df['Date'][i])
            df[col_name] = [1 if j == i else 0 for j in range(len(df))]
            missing_cols.append(col_name)

    # Fit the ARIMA model using pmdarima's auto_arima
    model = auto_arima(
        df['Revenue Losses'], 
        X=df[outlier_cols + missing_cols], 
        seasonal=True, m=7, 
        suppress_warnings=True
    )
    model.fit(df['Revenue Losses'], X=df[outlier_cols + missing_cols])

    # Forecast future values
    forecast_steps = 30
    forecast_df = pd.DataFrame(0, index=np.arange(forecast_steps), columns=outlier_cols + missing_cols)
    forecast_values, conf_int = model.predict(n_periods=forecast_steps, X=forecast_df, return_conf_int=True)
    
    # Store forecast and confidence intervals
    forecast_df['yhat'] = forecast_values
    forecast_df['yhat_lower'] = conf_int[:, 0]
    forecast_df['yhat_upper'] = conf_int[:, 1]

    # Undo Box-Cox transformation
    forecast_df['yhat'] = inv_boxcox(forecast_df['yhat'], lam)
    forecast_df['yhat_lower'] = inv_boxcox(forecast_df['yhat_lower'], lam)
    forecast_df['yhat_upper'] = inv_boxcox(forecast_df['yhat_upper'], lam)
    df['Revenue Losses'] = inv_boxcox(df['Revenue Losses'], lam)

    # Create new date column for forecast period
    forecast_df['Date'] = [df['Date'].max() + pd.Timedelta(days=i) for i in range(1, forecast_steps + 1)]

    # Plotting actual data and forecasts using Plotly
    actual_trace = go.Scatter(
        x=orig_df['Date'],
        y=orig_df['Revenue Losses'],
        mode='lines+markers',
        name='Actual Revenue Losses'
    )
    forecast_trace = go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['yhat'],
        mode='lines+markers',
        name='Forecast'
    )
    interval_trace = go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'][::-1].tolist(),
        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 100, 80, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Prediction Interval'
    )
    
    fig = go.Figure(data=[actual_trace, interval_trace, forecast_trace])
    fig.update_layout(title='Revenue Losses and Forecast', template='none')

    # Save plot as HTML to temporary Lambda storage
    html_output = '/tmp/Daily_ARIMA_Forecast.html'
    with open(html_output, 'w') as f:
        f.write(fig.to_html(full_html=True))

    # Upload HTML to S3 and make it publicly accessible
    s3.upload_file(html_output, bucket, 'Daily_ARIMA_Forecast.html', ExtraArgs={'ContentType': 'text/html', 'ACL': 'public-read'})

    # Return the S3 link to the uploaded forecast
    return {
        'statusCode': 200,
        'body': f"Forecast saved to: https://{bucket}.s3.amazonaws.com/Daily_ARIMA_Forecast.html"
    }
