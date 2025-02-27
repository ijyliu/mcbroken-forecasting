import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.special import inv_boxcox
from pmdarima import auto_arima
import plotly.graph_objs as go
import time

def arima_forecast(df):
    # Start timing
    start_time = time.time()
    print(f"ARIMA forecast started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    # Track data preparation time
    data_prep_time = time.time()
    print(f"Data preparation completed in: {(data_prep_time - start_time)/60:.2f} minutes")

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
    
    # Track feature creation time
    feature_time = time.time()
    print(f"Feature creation completed in: {(feature_time - data_prep_time)/60:.2f} minutes")
    print(f"Created {len(outlier_cols)} outlier features and {len(missing_cols)} missing features")

    # Fit the ARIMA model using pmdarima's auto_arima
    print("Starting ARIMA model fitting...")
    model = auto_arima(
        df['Revenue Losses'], 
        X=df[outlier_cols + missing_cols], 
        seasonal=True, m=7, 
        suppress_warnings=True,
        stepwise=True,
        n_jobs=-1
    )
    model.fit(df['Revenue Losses'], X=df[outlier_cols + missing_cols])
    
    # Track model fitting time
    model_time = time.time()
    print(f"ARIMA model fitting completed in: {(model_time - feature_time)/60:.2f} minutes")
    print(f"Final ARIMA model: {model.summary()}")

    # Forecast future values
    print("Generating forecasts...")
    forecast_steps = 30
    forecast_df = pd.DataFrame(0, index=np.arange(forecast_steps), columns=outlier_cols + missing_cols)
    forecast_values, conf_int = model.predict(n_periods=forecast_steps, X=forecast_df, return_conf_int=True)
    
    # Track forecasting time
    forecast_time = time.time()
    print(f"Forecast generation completed in: {(forecast_time - model_time)/60:.2f} minutes")
    
    # Store forecast and confidence intervals
    forecast_df['yhat'] = list(forecast_values)
    forecast_df['yhat_lower'] = list(conf_int[:, 0])
    forecast_df['yhat_upper'] = list(conf_int[:, 1])

    # Undo Box-Cox transformation
    forecast_df['yhat'] = inv_boxcox(forecast_df['yhat'], lam)
    forecast_df['yhat_lower'] = inv_boxcox(forecast_df['yhat_lower'], lam)
    forecast_df['yhat_upper'] = inv_boxcox(forecast_df['yhat_upper'], lam)
    df['Revenue Losses'] = inv_boxcox(df['Revenue Losses'], lam)

    # Create new date column for forecast period
    forecast_df['Date'] = [df['Date'].max() + pd.Timedelta(days=i) for i in range(1, forecast_steps + 1)]

    # Track transformation time
    transform_time = time.time()
    print(f"Inverse transformations completed in: {(transform_time - forecast_time)/60:.2f} minutes")

    return model, forecast_df, orig_df

def plot_forecast(forecast_df, orig_df):

    # Start timing
    start_time = time.time()

    # Limit orig_df to last 90 days for plot
    orig_df = orig_df.sort_values('Date').tail(90)

    # Plotting actual data and forecasts using Plotly
    print("Creating visualization...")
    actual_trace = go.Scatter(
        x=orig_df['Date'],
        y=orig_df['Revenue Losses'],
        mode='lines+markers',
        name='Actual Losses',
        hoverinfo='text',
        hovertext=[f"   {date.strftime('%a, %b %d, %Y')}   <br>   Actual Losses   <br>   ${revenue:,.0f}   " for date, revenue in zip(orig_df['Date'], orig_df['Revenue Losses'])],
        # original data muted blue: #1f77b4
        line=dict(color='#1f77b4'), 
        legendrank=1
    )
    
    forecast_trace = go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['yhat'],
        mode='lines+markers',
        name='ARIMA Forecast',
        hoverinfo='text',
        hovertext=[f"   {date.strftime('%a, %b %d, %Y')}   <br>   ARIMA Forecast   <br>   ${revenue:,.0f}   " for date, revenue in zip(forecast_df['Date'], forecast_df['yhat'])],
        # arima muted orange: #ff7f0e
        line=dict(dash='solid', color='#ff7f0e'),
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
