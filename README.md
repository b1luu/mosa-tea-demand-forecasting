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
## Next Steps

Planned extensions include:
- Visualization of revenue and weather relationships
- Regression and tree-based models to quantify weather effects
- Short-term demand forecasting using weather as exogenous features