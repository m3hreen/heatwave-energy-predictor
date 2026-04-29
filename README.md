# Heatwave Energy Demand Predictor

A machine learning project that predicts **heatwave-driven electricity demand spikes** using historical weather and grid demand data, helping identify high-risk periods and support smarter, more resilient energy planning.

## Demo & Live Project

- 🎥 **Demo Video:** [Watch our 3-minute walkthrough](https://youtu.be/UQu0TCSYoWI)  
- 🚀 **Live Zerve Project:** [Explore the full analysis](https://app.zerve.ai/notebook/5b20b858-5710-4802-ad05-57648709dfe2)

## Overview
As heatwaves become more frequent and intense, electricity grids face growing stress from surges in air conditioning demand. This project explores whether historical weather patterns can be used to predict when extreme heat may push electricity demand into dangerous territory.

Using weather data from Open-Meteo and electricity demand data from California’s CAISO grid (via the EIA API), we built a predictive model that forecasts demand and classifies periods into **Low, Medium, and High Risk** based on grid stress.

---

## Features
- Predicts hourly electricity demand during heatwave conditions  
- Classifies grid stress into Low, Medium, and High risk  
- Exploratory visualizations revealing demand patterns  
- Feature importance analysis for model interpretability  
- Focus on climate resilience and energy planning applications  

---

## Tech Stack
- Python  
- Jupyter Notebook  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Random Forest Regressor  
- Open-Meteo API  
- U.S. EIA API (CAISO demand data)

---

## Dataset
### Weather Data
Hourly weather observations (2022–2025):
- Temperature  
- Humidity  
- Apparent ("Feels Like") Temperature

Source: Open-Meteo API

### Grid Demand Data
Hourly electricity demand across California's CAISO grid.

Source: U.S. Energy Information Administration (EIA) API

---

## Methodology

### Feature Engineering
The model uses seven input features:

- Temperature  
- Humidity  
- Apparent Temperature  
- Hour of Day  
- Month  
- Heatwave Indicator (≥32°C)  
- Weekend Flag  

### Model
A **Random Forest Regressor** was trained using an 80/20 train-test split.

### Risk Classification
Predicted demand is categorized into:

- **Low Risk** — Below 70th percentile  
- **Medium Risk** — 70th–90th percentile  
- **High Risk** — Above 90th percentile  

---

## Key Findings
- Electricity demand follows a **U-shaped relationship** with temperature.
- Heatwave days show elevated average demand.
- **Hour of day and seasonality** were stronger predictors than temperature alone.
- Weak linear correlations suggested machine learning was needed to capture nonlinear effects.

---

## Why It Matters
Extreme heat events are increasing due to climate change, creating growing challenges for electricity infrastructure.

This project demonstrates how machine learning can support:

- Demand forecasting  
- Grid resilience  
- Early warning systems  
- Smarter energy planning

---

## Future Improvements
- Add more weather variables (solar radiation, wind, wildfire smoke)
- Explore advanced models
- Expand beyond California
- Integrate real-time demand risk monitoring

---


## Screenshots

### 1. Temperature vs Electricity Demand
Illustrates the U-shaped relationship between temperature and demand.

![Temperature vs Demand](/screenshots/temp-vs-demand.png)

---

### 2. Heatwave vs Normal Day Demand
Comparison of average electricity demand on heatwave days versus non-heatwave days, highlighting increased grid stress during extreme temperatures.

![Demand by Hour](/screenshots/heatwave-vs-normal-demand.png)

---

### 3. Feature Importance Analysis
Highlights which variables contribute most to demand prediction.

![Feature Importance](/screenshots/feature_importance_visual.png)

---

### 4. Correlation Heatmap (Weather Features vs Demand)
Weak linear correlations suggest machine learning is needed to capture complex demand patterns.

![Risk Prediction Output](/screenshots/correlation-map.png)

---

### 5. Data Cleaning and Modeling Workflow
Jupyter notebook workflow showing data preprocessing, cleaning, and model development pipeline.

![Notebook Workflow](/screenshots/heatwave-risk-zerveai.png)

---

## Authors
- Shimza Warraich
- Mehreen Morshed
- Muskan Morshed
