
import boto3
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.special import inv_boxcox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objs as go
from io import BytesIO

def lambda_handler(event, context):
    # S3 details
    s3 = boto3.client('s3')
    bucket = 'mcbroken-daily'  # S3 bucket name
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

    # Set Outlier values to missing
    df.loc[df['Outlier'], 'Revenue Losses'] = None
    # While loop to fill with 7 day lags
    # Alternate between computing 7 day lags
    # and filling missing values with 7 day lags
    iterations = 0
    while df['Revenue Losses'].isnull().sum() > 0 and iterations < len(df):
        # Compute 7 day lags
        df['7-day lag'] = df['Revenue Losses'].shift(7)
        # Fill missing values with 7 day lags
        df['Revenue Losses'] = df['Revenue Losses'].fillna(df['7-day lag'])
        # Increment iterations
        iterations += 1
    # For any remaining items, back fill with 7 days forward
    iterations = 0
    while df['Revenue Losses'].isnull().sum() > 0 and iterations < len(df):
        # Compute 7 days forward
        df['7-day forward'] = df['Revenue Losses'].shift(-7)
        # Fill missing values with 7 days forward
        df['Revenue Losses'] = df['Revenue Losses'].fillna(df['7-day forward'])
        # Increment iterations
        iterations += 1
    # For any remaining items, fill with mean of the data
    df['Revenue Losses'] = df['Revenue Losses'].fillna(df['Revenue Losses'].mean())

    df = df[['Date', 'Revenue Losses']]

    # Holt-Winters' Method
    model_hw = ExponentialSmoothing(df['Revenue Losses'], 
                                    trend="add", 
                                    damped_trend=True,
                                    seasonal="add", 
                                    seasonal_periods=7, 
                                    initialization_method="estimated")
    fit_hw = model_hw.fit()
    forecast_steps = 30
    forecast_values = fit_hw.forecast(steps=forecast_steps)

    # 1. Get Residuals from the Training Data
    train_residuals = df['Revenue Losses'] - fit_hw.fittedvalues

    # 2. Bootstrapping Function
    def bootstrap_forecast(train_data, train_residuals, test_len, n_bootstraps=500):
        forecasts = []
        for _ in range(n_bootstraps):
            # 2a. Resample Residuals with Replacement
            resampled_residuals = list(resample(train_residuals, replace=True))

            # 2b. Create "Bootstrapped" Training Data
            bootstrapped_train = train_data + resampled_residuals

            # 2c. Refit the Model on Bootstrapped Data
            boot_model = ExponentialSmoothing(bootstrapped_train,                                 
                                    trend="add", 
                                    damped_trend=True,
                                    seasonal="add", 
                                    seasonal_periods=7, 
                                    initialization_method="estimated")
            boot_model_fit = boot_model.fit()

            # 2d. Generate Forecasts
            boot_forecast = boot_model_fit.forecast(test_len)
            forecasts.append(boot_forecast)

        return np.array(forecasts)  # Return as a NumPy array for easier manipulation

    # 3. Generate Bootstrapped Forecasts
    n_bootstraps = 500  # Number of bootstrap iterations (adjust as needed)
    boot_forecasts = bootstrap_forecast(df['Revenue Losses'], train_residuals, forecast_steps, n_bootstraps)

    # 4. Calculate Prediction Intervals
    alpha = 0.05  # Significance level (for 95% interval)
    lower_bounds = np.percentile(boot_forecasts, (alpha / 2) * 100, axis=0)  # Lower percentile
    upper_bounds = np.percentile(boot_forecasts, (1 - alpha / 2) * 100, axis=0)  # Upper percentile

    # Store forecast and confidence intervals
    forecast_df = pd.DataFrame({
        'yhat': forecast_values,
        'yhat_lower': lower_bounds,
        'yhat_upper': upper_bounds,
        'Date': [df['Date'].max() + pd.Timedelta(days=i) for i in range(1, forecast_steps + 1)]
    })

    # Undo Box-Cox transformation
    forecast_df['yhat'] = inv_boxcox(forecast_df['yhat'], lam)
    forecast_df['yhat_lower'] = inv_boxcox(forecast_df['yhat_lower'], lam)
    forecast_df['yhat_upper'] = inv_boxcox(forecast_df['yhat_upper'], lam)
    df['Revenue Losses'] = inv_boxcox(df['Revenue Losses'], lam)

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
    html_output = '/tmp/Daily_ETS_Forecast.html'
    with open(html_output, 'w') as f:
        f.write(fig.to_html(full_html=True))

    # Upload HTML to S3 and make it publicly accessible
    s3.upload_file(html_output, bucket, 'Daily_ETS_Forecast.html', ExtraArgs={'ContentType': 'text/html', 'ACL': 'public-read'})

    # Return the S3 link to the uploaded forecast
    return {
        'statusCode': 200,
        'body': f"Forecast saved to: https://{bucket}.s3.amazonaws.com/Daily_ETS_Forecast.html"
    }

