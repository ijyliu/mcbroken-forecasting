import json
import boto3
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime
import pytz

def lambda_handler(event, context):
    # Fetch the data (from DynamoDB or GitHub)
    data = fetch_data()

    # Run basic forecasting (e.g., average)
    forecast = run_basic_forecast(data)

    # Create Plotly visualization
    fig = create_plotly_figure(forecast)

    # Save to S3 or prepare for Google Sites
    save_visualization(fig)

    return {
        'statusCode': 200,
        'body': json.dumps('Forecasting completed successfully!')
    }

def fetch_data():
    # Example: If fetching data from GitHub (raw CSV URL)
    url = "https://raw.githubusercontent.com/user/repo/main/data.csv"
    df = pd.read_csv(url)
    return df

def run_basic_forecast(data):
    # Simple model: forecast as the average of the data
    forecast = data['value'].mean()
    return forecast

def create_plotly_figure(forecast):
    # Create a simple Plotly figure
    fig = px.line(x=[1, 2, 3], y=[forecast, forecast, forecast], title="Daily Forecast")
    return fig

def save_visualization(fig):
    # Save the Plotly figure to an S3 bucket (or prepare for Google Sites)
    fig.write_html("/tmp/forecast.html")  # Save as HTML locally in Lambda

    # Upload to S3 (ensure your Lambda has the right permissions)
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/forecast.html', 'your-s3-bucket', 'forecast/forecast.html')
