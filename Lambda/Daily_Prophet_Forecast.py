# Import necessary libraries
import pandas as pd
import scipy.stats as stats
from scipy.special import inv_boxcox
from prophet import Prophet
import optuna
from sklearn.metrics import root_mean_squared_error
import plotly.graph_objs as go
import boto3
from io import BytesIO

# Set up S3 client and specify bucket and key
s3 = boto3.client('s3')
bucket = 'mcbroken-daily'  # S3 bucket name
key = 'Clean_McBroken_Daily.xlsx'  # File name in S3

def lambda_handler(event, context):
    # Load data from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_excel(BytesIO(obj['Body'].read()))

    # Data preprocessing: Selecting relevant columns and renaming them for Prophet
    df = df[['Date', 'Revenue Losses', 'Train', 'Outlier']].rename(columns={'Revenue Losses': 'y', 'Date': 'ds'})
    df = df.sort_values('ds').reset_index(drop=True)
    df = df[-1582:]  # Keep the last 1582 records for modeling

    # Box-Cox Transformation on non-outlier and non-missing data to stabilize variance
    orig_df = df.copy() # save original df for later plots
    no_out_missing = df[(df['Outlier'] == False) & (df['y'].notnull())]
    no_out_missing['y'], lam = stats.boxcox(no_out_missing['y'])
    df.loc[no_out_missing.index, 'y'] = no_out_missing['y']

    # Create dummy variables for outliers to improve model accuracy
    outlier_cols = []
    for i in range(len(df)):
        if df.loc[i, 'Outlier'] == 1:
            # Create a dummy variable for each outlier date
            df['Outlier_' + str(df['ds'][i])] = [1 if j == i else 0 for j in range(len(df))]
            outlier_cols.append('Outlier_' + str(df['ds'][i]))

    # Objective function for hyperparameter optimization using Optuna
    def objective(trial):
        # Define hyperparameter search space
        param_grid = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0),
            "changepoint_range": trial.suggest_float("changepoint_range", 0.5, 1.0)
        }

        # Create Prophet model with trial parameters
        m = Prophet(**param_grid)
        for col in outlier_cols:
            m.add_regressor(col)  # Add outlier dummies as regressors

        # Split data into training and validation sets
        obj_df_train = df[:-30]  # Training set excluding the last 30 days
        obj_df_val = df[-30:]    # Validation set (last 30 days)

        # Fit model and make predictions
        m.fit(obj_df_train)
        forecast = m.predict(obj_df_val)

        # Calculate RMSE as the optimization objective
        rmse = root_mean_squared_error(obj_df_val['y'], forecast['yhat'], squared=False)
        return rmse

    # Hyperparameter tuning using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=1800)

    # Extract best hyperparameters from Optuna study
    optimal_changepoint_prior = study.best_params['changepoint_prior_scale']
    optimal_seas_prior = study.best_params['seasonality_prior_scale']
    optimal_changepoint_range = study.best_params['changepoint_range']

    # Train final model using the optimized hyperparameters
    m = Prophet(
        interval_width=0.95,
        seasonality_prior_scale=optimal_seas_prior,
        changepoint_prior_scale=optimal_changepoint_prior,
        changepoint_range=optimal_changepoint_range
    )
    for col in outlier_cols:
        m.add_regressor(col)

    # Fit model on the full dataset
    m.fit(df)

    # Forecasting for the next 30 days
    future = m.make_future_dataframe(periods=30).tail(30)
    for col in outlier_cols:
        future[col] = [0] * len(future)  # Set future outlier dummies to 0

    forecast = m.predict(future)

    # Inverse Box-Cox transformation for interpretability
    forecast['yhat'] = inv_boxcox(forecast['yhat'], lam)
    forecast['yhat_lower'] = inv_boxcox(forecast['yhat_lower'], lam)
    forecast['yhat_upper'] = inv_boxcox(forecast['yhat_upper'], lam)
    df['y'] = inv_boxcox(df['y'], lam)

    # Create new date column for forecast period
    forecast['Date'] = [df['ds'].max() + pd.Timedelta(days=i) for i in range(1, 31)]

    # Plotting actual data and forecasts using Plotly
    actual_trace = go.Scatter(
        x=orig_df['ds'],
        y=orig_df['y'],
        mode='lines+markers',
        name='Actual Revenue Losses'
    )

    forecast_trace = go.Scatter(
        x=forecast['Date'],
        y=forecast['yhat'],
        mode='lines+markers',
        name='Forecast'
    )

    interval_trace = go.Scatter(
        x=forecast['Date'].tolist() + forecast['Date'][::-1].tolist(),
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 100, 80, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Prediction Interval'
    )

    fig = go.Figure(data=[actual_trace, interval_trace, forecast_trace])
    fig.update_layout(
        title='Revenue Losses and Forecast',
        template='none'
    )

    # Save plot as HTML and upload to S3 for sharing
    html_output = 'Daily_Prophet_Forecast.html'
    with open('/tmp/' + html_output, 'w') as f:
        f.write(fig.to_html(full_html=True))

    # Upload HTML to S3 and make it publicly accessible
    s3.upload_file('/tmp/' + html_output, bucket, html_output, ExtraArgs={'ContentType': 'text/html', 'ACL': 'public-read'})

    # Return the S3 link to the uploaded forecast
    return {
        'statusCode': 200,
        'body': f"Forecast saved to: https://{bucket}.s3.amazonaws.com/{html_output}"
    }
