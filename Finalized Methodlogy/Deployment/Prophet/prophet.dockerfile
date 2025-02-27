# Base image for Lambda with Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Working directory
WORKDIR /var/task

# Copy Lambda function and requirements
COPY Deployment/Prophet/Prophet_Lambda.py .
COPY Deployment/Prophet/requirements.txt .

# Copy necessary source files
COPY Source/Prophet_Forecast_Functions.py .
COPY Source/Load_Data_AWS.py .

# Install dependencies
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install -r requirements.txt

# Set the handler
CMD ["Prophet_Lambda.lambda_handler"] 
