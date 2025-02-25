# Base image for Lambda with Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Copy function code and requirements.txt
COPY app/ /var/task/
COPY requirements.txt .

# Install dependencies
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install -r requirements.txt

# Set the handler
CMD ["app.lambda_handler"]
