
from Daily_ETS_Forecast import *

def lambda_handler(event, context):
    
    # Run forecast function
    status_json = aws_run()

    # Return status json
    return status_json
