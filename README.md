# McBroken Forecasting

Time Series Forecasting of McDonald's Ice Cream Machine Outages

Isaac Liu

![McBroken Forecast Example](image/README/mcbroken_forecast_example.png)

## Project Overview

This project implements and compares three different time series forecasting methods to predict revenue losses from McDonald's ice cream machine outages. Using data from mcbroken.com, these models predict future outages and associated revenue impacts, estimated at $625 per machine-day of downtime.

The forecasting models include:
- **Prophet**: Facebook's time series forecasting library with advanced seasonality modeling (Best performer with ~7% MAPE)
- **ARIMA**: Auto-Regressive Integrated Moving Average model with seasonal components
- **ETS**: Exponential Smoothing State Space model with trend and seasonality

Each implementation features hyperparameter optimization, outlier handling, data transformations, and confidence interval generation. The models produce interactive visualizations with detailed forecasts for the next 30 days.

Live daily forecasts are hosted on my [Google Sites page](https://sites.google.com/view/isaac-liu) along with detailed methodology writeups and analysis.

## Repository Structure

```
mcbroken-forecasting/
├── Experimental Notebooks/       # Development and testing notebooks for each method
├── Lambda/                       # AWS Lambda deployment files
│   ├── ARIMA/                    # ARIMA model implementation
│   ├── ETS/                      # ETS model implementation  
│   └── Prophet/                  # Prophet model implementation
├── Testing/                      # Local testing outputs
└── environment.yml               # Conda environment specification
```

## Technical Implementation

The forecasting pipeline combines:
1. **Data Processing**: Automated data collection from McBroken API, cleaning, and storage in AWS S3/DynamoDB
2. **Model Training**: Each method implements specialized handling of seasonality, outliers, and missing values
3. **Hyperparameter Optimization**: Using techniques like Optuna for Prophet to optimize model parameters
4. **Visualization**: Interactive Plotly visualizations with forecast values, confidence intervals, and rich hover information
5. **Deployment**: Serverless deployment via AWS Lambda with Docker containers for each model

## Technologies

- **Python**
  - pandas, numpy, scipy
  - pmdarima (for ARIMA modeling)
  - Prophet
  - statsmodels (for ETS modeling)
  - Optuna (hyperparameter optimization)
  - Plotly (interactive visualizations)
  - scikit-learn (metrics)
  - boto3 (AWS integration)
  
- **Cloud & Deployment**
  - AWS Lambda
  - Amazon S3
  - Amazon DynamoDB
  - Docker containers
  
- **Development Environment**
  - Conda/Mamba for environment management
  - Jupyter notebooks for experimentation

## Results

The Prophet model achieved the best performance with approximately 7% Mean Absolute Percentage Error (MAPE) on out-of-sample testing, outperforming both ARIMA and ETS implementations. Visualizations include the actual historical values, forecasted values, and 95% confidence intervals.

## Usage

### Local Testing
```bash
# Activate the conda environment
conda activate mcbroken-env

# Run forecast for Prophet model
cd Lambda/Prophet
python Daily_Prophet_Forecast.py

# Run forecast for ARIMA model
cd Lambda/ARIMA
python Daily_ARIMA_Forecast.py

# Run forecast for ETS model
cd Lambda/ETS
python Daily_ETS_Forecast.py
```

### AWS Deployment
Each forecasting method is containerized in its respective Dockerfile and deployed as AWS Lambda functions for automated daily forecasting.

## Future Work

- Improved handling of extreme outliers
- Ensemble methods combining all three forecasting approaches
- Incorporation of exogenous variables like weather data and holidays
- Real-time notifications for significant expected increases in machine outages 
