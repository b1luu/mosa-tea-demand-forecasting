# Mosa Tea Demand Forecasting

This project analyzes and forecasts daily revenue for Mosa Tea by combining
transaction-level order data with external weather data from NOAA.

The pipeline aggregates raw order data into daily revenue, integrates
San Diego daily weather (average temperature and precipitation), and produces
analysis-ready datasets for exploratory data analysis and modeling.

## Data Pipeline Overview

1. **Order Processing**
   - Raw order-level data is cleaned and anonymized.
   - Orders are aggregated to daily revenue with weekday indicators.

2. **Weather Integration**
   - Daily weather data is sourced from NOAA (San Diego International Airport).
   - Features include daily average temperature and total precipitation.
   - Weather data is cleaned and standardized for merging.

3. **Data Merging**
   - Daily revenue is merged with daily weather on date.
   - Final dataset is saved to `data/analytics/daily_revenue_weather_v1.csv`.

## Exploratory Analysis

Initial exploratory analysis summarizes:
- Revenue distribution and variability
- Average revenue by day of week
- Differences in revenue between rainy and non-rainy days

Summary tables are generated using:
- `src/summarize_revenue_weather.py`

## Regression + Visualizations

The summary script also fits a simple linear regression to estimate the
association between weather and revenue while controlling for weekday effects.
This is **not causal**, but it provides directional signals and effect sizes.

Outputs saved to `data/analytics/`:
- `regression_weather_coefficients_v1.csv`
- `regression_weather_metrics_v1.csv`

If `matplotlib` is installed, plots are saved to `data/analytics/figures/`:
- `regression_coefficients_v1.png`
- `regression_actual_vs_predicted_v1.png`
- `regression_residuals_vs_fitted_v1.png`
- `regression_residuals_hist_v1.png`
- `scatter_revenue_vs_temp_v1.png`
- `scatter_revenue_vs_rain_v1.png`

## How To Run

1) Build the daily revenue + weather dataset:
```
python .\src\make_daily_revenue_weather_v1.py
```

2) Generate summaries, regression, and plots:
```
python .\src\summarize_revenue_weather.py
```

Optional dependency for plots:
```
pip install matplotlib
```
