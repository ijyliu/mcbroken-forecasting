# Base image for Lambda with Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Copy function code and requirements.txt
COPY * .
# Copy functions from the Source folder
COPY ../Source/Prophet_Forecast_Functions.py .
COPY ../Source/Load_Data_AWS.py .

# Install dependencies
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install -r requirements.txt

# Set the handler
CMD ["Prophet_Lambda.lambda_handler"]
