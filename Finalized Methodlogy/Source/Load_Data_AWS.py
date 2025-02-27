import pandas as pd
import boto3
from io import BytesIO

def load_data_aws():
    """
    Load the Clean_McBroken_Daily.xlsx file from the S3 bucket.
    
    Returns:
        pandas.DataFrame: The loaded data
    """
    # S3 details
    s3 = boto3.client('s3')
    bucket = 'mcbroken-bucket'  # S3 bucket name
    key = 'Clean_McBroken_Daily.xlsx'  # File name in S3

    # Load data from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_excel(BytesIO(obj['Body'].read()))

    return df 
