# Base image for Lambda with Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Working directory
WORKDIR /var/task

# Copy Lambda function and requirements
COPY Deployment/Daily_Data_Cleaning/Data_Cleaning_Lambda.py .
COPY Deployment/Daily_Data_Cleaning/requirements.txt .

# Copy necessary source files
COPY Source/Data_Cleaning_Functions.py .
COPY Source/Load_Data_AWS.py .

# Install dependencies
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install -r requirements.txt

# Set the handler
CMD ["Data_Cleaning_Lambda.lambda_handler"] 
