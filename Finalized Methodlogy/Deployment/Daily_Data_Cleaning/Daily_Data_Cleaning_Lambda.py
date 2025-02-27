import boto3
import io
from Data_Cleaning_Functions import *

def lambda_handler(event, context):

    # Run data preparation
    df = concat_data()
    df = clean_data(df)
    
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
