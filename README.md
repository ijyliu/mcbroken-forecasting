# McBroken-Forecasting

Time Series Forecasting of McDonald's Ice Cream Machine Outages

Isaac Liu

![McBroken Forecast Example](image/README/mcbroken_forecast_example.png)

This project implements and compares three different time series forecasting methods to predict revenue losses from McDonald's ice cream machine outages. Using data from mcbroken.com, these models predict future outages and associated revenue impacts, estimated at $625 per machine-day of downtime.

The forecasting models include:
- **Prophet**: Facebook's time series forecasting library with advanced seasonality modeling (Best performer with ~7% MAPE)
- **ARIMA**: Auto-Regressive Integrated Moving Average model with seasonal components
- **ETS**: Exponential Smoothing State Space model with trend and seasonality

Each implementation features hyperparameter optimization, outlier handling, data transformations, and confidence interval generation. The models produce interactive visualizations with detailed forecasts for the next 30 days. The repository is organized with Experimental Notebooks for development, Lambda deployment folders for each model implementation (ARIMA, ETS, Prophet), and a Testing directory for local outputs.

The forecasting pipeline combines automated data collection from McBroken API with cleaning and storage in AWS S3/DynamoDB. Each method implements specialized handling of seasonality, outliers, and missing values, with techniques like Optuna used for hyperparameter optimization. The Prophet model achieved the best performance with approximately 7% Mean Absolute Percentage Error (MAPE) on out-of-sample testing, outperforming both ARIMA and ETS implementations.

For local testing, activate the conda environment (`conda activate mcbroken-env`) and run the Python scripts in each model's directory. For deployment, each forecasting method is containerized and deployed as AWS Lambda functions for automated daily forecasting. Future plans include improved handling of extreme outliers, ensemble methods combining the approaches, incorporation of exogenous variables, and real-time notifications for significant expected increases.

Live daily forecasts are hosted on my [Google Sites page](https://sites.google.com/view/isaac-liu) along with detailed methodology writeups and analysis.

## Technologies (not exhaustive!)

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
