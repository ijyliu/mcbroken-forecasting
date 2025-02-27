# Import necessary libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.special import inv_boxcox
from prophet import Prophet
import optuna
#from sklearn.metrics import root_mean_squared_error
import plotly.graph_objs as go
import time

def prophet_forecast(df):
    # Start timing
    start_time = time.time()
    print(f"Prophet forecast started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data preprocessing: Selecting relevant columns and renaming them for Prophet
    df = df[['Date', 'Revenue Losses', 'Outlier']].rename(columns={'Revenue Losses': 'y', 'Date': 'ds'})
    df = df.sort_values('ds').reset_index(drop=True)
    df = df[-1582:]  # Keep the last 1582 records for modeling
    # Reset index
    df = df.reset_index(drop=True)

    # Box-Cox Transformation on non-outlier and non-missing data to stabilize variance
    orig_df = df.copy() # save original df for later plots
    no_out_missing = df[(df['Outlier'] == False) & (df['y'].notnull())]
    # Print any cases where y is not positive
    print(no_out_missing[no_out_missing['y'] <= 0])
    no_out_missing['y'], lam = stats.boxcox(no_out_missing['y'])
    df.loc[no_out_missing.index, 'y'] = no_out_missing['y']
    
    # Track data preparation time
    data_prep_time = time.time()
    print(f"Data preparation completed in: {(data_prep_time - start_time)/60:.2f} minutes")

    # Create dummy variables for outliers to improve model accuracy
    outlier_cols = []
    for i in range(len(df)):
        if df.loc[i, 'Outlier'] == 1:
            # Create a dummy variable for each outlier date
            df['Outlier_' + str(df['ds'][i])] = [1 if j == i else 0 for j in range(len(df))]
            outlier_cols.append('Outlier_' + str(df['ds'][i]))
    
    # Track feature creation time
    feature_time = time.time()
    print(f"Feature creation completed in: {(feature_time - data_prep_time)/60:.2f} minutes")
    print(f"Created {len(outlier_cols)} outlier features")

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
        rmse = np.sqrt(np.nanmean((np.array(obj_df_val['y']) - np.array(forecast['yhat'])) ** 2))
        
        return rmse
    
    # Hyperparameter tuning using Optuna
    print("Starting hyperparameter optimization with Optuna...")
    
    # Create study 
    study = optuna.create_study(direction="minimize")
    
    # Run optimization with max 100 trials and timeout set to 10 minutes (600 seconds)
    print(f"Optimization will run for up to 100 trials or 10 minutes")
    study.optimize(objective, n_trials=100, timeout=600)
    
    # Track optimization time
    optim_time = time.time()
    print(f"Hyperparameter optimization completed in: {(optim_time - feature_time)/60:.2f} minutes")
    print(f"Best parameters: {study.best_params}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Number of completed trials: {len(study.trials)}")

    # Extract best hyperparameters from Optuna study
    optimal_changepoint_prior = study.best_params['changepoint_prior_scale']
    optimal_seas_prior = study.best_params['seasonality_prior_scale']
    optimal_changepoint_range = study.best_params['changepoint_range']

    # Train final model using the optimized hyperparameters
    print("Fitting final Prophet model...")
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

    # Fitted values
    fitted_values = m.predict(df)
    
    # Track model fitting time
    model_time = time.time()
    print(f"Final model fitting completed in: {(model_time - optim_time)/60:.2f} minutes")

    # Forecasting for the next 30 days
    print("Generating forecasts...")
    future = m.make_future_dataframe(periods=30).tail(30)
    for col in outlier_cols:
        future[col] = [0] * len(future)  # Set future outlier dummies to 0

    forecast = m.predict(future)
    
    # Track forecasting time
    forecast_time = time.time()
    print(f"Forecast generation completed in: {(forecast_time - model_time)/60:.2f} minutes")

    # Inverse Box-Cox transformation for interpretability
    forecast['yhat'] = inv_boxcox(forecast['yhat'], lam)
    forecast['yhat_lower'] = inv_boxcox(forecast['yhat_lower'], lam)
    forecast['yhat_upper'] = inv_boxcox(forecast['yhat_upper'], lam)
    df['y'] = inv_boxcox(df['y'], lam)
    fitted_values['yhat'] = inv_boxcox(fitted_values['yhat'], lam)
    fitted_values['yhat_lower'] = inv_boxcox(fitted_values['yhat_lower'], lam)
    fitted_values['yhat_upper'] = inv_boxcox(fitted_values['yhat_upper'], lam)

    # Create new date column for forecast period
    forecast['Date'] = [df['ds'].max() + pd.Timedelta(days=i) for i in range(1, 31)]

    # Track transformation time
    transform_time = time.time()
    print(f"Inverse transformations completed in: {(transform_time - forecast_time)/60:.2f} minutes")

    return m, forecast, orig_df, fitted_values

def plot_forecast(forecast_df, orig_df):

    # Start timing
    start_time = time.time()

    # Limit orig_df to last 90 days for plot
    orig_df = orig_df.sort_values('ds').tail(90)
    
    # Plotting actual data and forecasts using Plotly
    print("Creating visualization...")
    actual_trace = go.Scatter(
        x=orig_df['ds'],
        y=orig_df['y'],
        mode='lines+markers',
        name='Actual Losses',
        hoverinfo='text',
        hovertext=[f"   {date.strftime('%a, %b %d, %Y')}   <br>   Actual Losses   <br>   ${revenue:,.0f}   " for date, revenue in zip(orig_df['ds'], orig_df['y'])],
        # original data muted blue: #1f77b4
        line=dict(color='#1f77b4'), 
        legendrank=1
    )

    forecast_trace = go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['yhat'],
        mode='lines+markers',
        name='Prophet Forecast',
        hoverinfo='text',
        hovertext=[f"   {date.strftime('%a, %b %d, %Y')}   <br>   Prophet Forecast   <br>   ${revenue:,.0f}   " for date, revenue in zip(forecast_df['Date'], forecast_df['yhat'])],
        # prophet red: #d62728
        line=dict(dash='solid', color='#d62728'),
        legendrank=2
    )

    # Compute trace color
    # Take standard color for forecast, edit alpha channel
    interval_color = 'rgba' + str(tuple(int(forecast_trace['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,))
    interval_trace = go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'][::-1].tolist(),
        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor=interval_color,
        line=dict(color=interval_color),
        name='95% Prediction Interval',
        hoverinfo='skip',
        legendrank=3
    )

    fig = go.Figure(data=[actual_trace, interval_trace, forecast_trace])
    
    # Update layout for elegance
    fig.update_layout(
        title='Revenue Losses from Broken McDonald\'s Ice Cream Machines ($625 per machine‑day)',
        template='none',
        #hovermode='x unified',
        legend=dict(
            orientation='h',  # Horizontal
            x=0.5,            # Centered
            xanchor='center', # Centered
            y=-0.2,           # Below the plot area
            yanchor='top',
            traceorder='normal'
        ),
        # Add hoverlabel styling for white background with centered black text
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_color="black",
            align="auto"
        ),
    )

    # Update y-axis for custom scaling
    fig.update_yaxes(
        tickprefix="$",
        tickformat=".2s",  # Use "M" for millions and "K" for thousands
        tickfont=dict(size=12)
    )

    # Update x-axis for date format without day of the week
    fig.update_xaxes(
        tickformat="%b %d, %Y",  # Month, day, and year (no day of the week)
        tickmode='auto',
        tickfont=dict(size=12)
    )
    
    # Track visualization time
    viz_time = time.time()
    print(f"Visualization created in: {(viz_time - start_time)/60:.2f} minutes")
    
    return fig
