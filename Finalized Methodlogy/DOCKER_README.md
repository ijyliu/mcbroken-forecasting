# Docker Compose for McBroken Forecasting

This directory contains Docker Compose configuration for building and running the McBroken forecasting services.

## Images

The setup creates the following Docker images:
- `mcb-prophet` - Prophet forecasting model
- `mcb-arima` - ARIMA forecasting model
- `mcb-exponential-smoothing` - Exponential Smoothing forecasting model
- `mcb-data-cleaning` - Data cleaning pipeline

## Usage

### Build all images

```bash
docker-compose build
```

### Build a specific image

```bash
# Build just the Prophet model
docker-compose build prophet-lambda

# Build just the ARIMA model
docker-compose build arima-lambda

# Build just the Exponential Smoothing model
docker-compose build exponential-smoothing-lambda

# Build just the Data Cleaning pipeline
docker-compose build data-cleaning-lambda
```

### Pushing to AWS ECR

If you need to push these images to AWS ECR, you'll need to tag and push them:

```bash
# Login to AWS ECR
aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com

# Tag the images
docker tag mcb-prophet:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mcb-prophet:latest
docker tag mcb-arima:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mcb-arima:latest
docker tag mcb-exponential-smoothing:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mcb-exponential-smoothing:latest
docker tag mcb-data-cleaning:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mcb-data-cleaning:latest

# Push the images
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mcb-prophet:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mcb-arima:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mcb-exponential-smoothing:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mcb-data-cleaning:latest
```

Replace `YOUR_ACCOUNT_ID` and `YOUR_REGION` with your AWS account ID and region.

## Notes

- The build context is set to the parent directory, so all path references in the Dockerfiles are relative to the `Finalized Methodlogy` directory.
- Each service has its own Dockerfile in its respective directory.
- To update a service, modify its source files and then rebuild the specific service. 
