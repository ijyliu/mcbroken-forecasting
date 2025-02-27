# Base image for Lambda with Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Working directory
WORKDIR /var/task

# Copy Lambda function and requirements
COPY Deployment/Exponential_Smoothing/Exponential_Smoothing_Lambda.py .
COPY Deployment/Exponential_Smoothing/requirements.txt .

# Copy necessary source files
COPY Source/Exponential_Smoothing_Forecast_Functions.py .
COPY Source/Load_Data_AWS.py .

# Install dependencies
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install -r requirements.txt

# Set the handler
CMD ["Exponential_Smoothing_Lambda.lambda_handler"] 
