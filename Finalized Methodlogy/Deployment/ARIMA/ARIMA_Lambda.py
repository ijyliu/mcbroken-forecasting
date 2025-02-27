import boto3
from io import BytesIO
from ARIMA_Forecast_Functions import *
from Load_Data_AWS import load_data_aws

def save_output_aws(client, bucket, fig):
    # Save plot as HTML to temporary Lambda storage
    html_output = '/tmp/Daily_ARIMA_Forecast.html'
    with open(html_output, 'w') as f:
        f.write(fig.to_html(full_html=True))

    # Upload HTML to S3 and make it publicly accessible
    client.upload_file(html_output, bucket, 'Daily_ARIMA_Forecast.html', ExtraArgs={'ContentType': 'text/html', 'ACL': 'public-read'})

    # Return the S3 link to the uploaded forecast
    return {
        'statusCode': 200,
        'body': f"Forecast saved to: https://{bucket}.s3.amazonaws.com/Daily_ARIMA_Forecast.html"
    }

# Function for AWS runs
def aws_run():    
    
    # Load data from S3
    df = load_data_aws()

    # Generate forecast
    model, forecast_df, orig_df = arima_forecast(df)

    # Plot forecast
    fig = plot_forecast(forecast_df, orig_df)

    # Save output to S3
    client = boto3.client('s3')
    bucket = 'mcbroken-bucket'  # S3 bucket name
    return save_output_aws(client, bucket, fig)

def lambda_handler(event, context):
    
    # Run forecast function
    status_json = aws_run()

    # Return status json
    return status_json 
