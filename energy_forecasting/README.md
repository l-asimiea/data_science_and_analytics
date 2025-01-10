Electricity Demand Forecasting
==============================

Project Overview
-------------------
This project aims to forecast electricity demand using advanced machine learning models and provide insights through a interactive dashboard. 
The focus was on leveraging historical data, incoporating temporal and lag features in building a model suitable for making accurate forecasts, empowering users such as energy providers to effectively allocate, manage and balance energy generation resource against demand.  

Objectives
------------
1. Develop an understanding of historical electricity demand to obtain useful features for a forecast modelling.
2. Explore and evaluate machine learning models to achieve high forecast accuracy, and low error metrics.
3. Build a user-friendly dashboard for technical and non-technical stakeholders to visualise historical electricity demand and forecasts to a maximum of 3 years.

Data Overview
-------------
- **Dataset**: Electricity demand data at 30 minutes intervals, including solar and wind generation features from Januray 2009 to October 2024 (15 years). Link:
- **Key Features**: The features of interest include:
    - Temporal features: Features such as hour, day, week, month and year extractable from the date as well as 1 year rolling mean
    - Lagged features: Short-term shifts (1 hour, 1 day, 1 week) and long-term shifts (1 year and 2 year)
    - Target varibale: Total System Demand, tsd 
Other variables reviewed in the dataset include, solar and wind capacities, wind and solar generations. 

Methodology
------------
To be converted to a flowchart. Data Processing and EDA, Feature Engineering and Model Training, Model Evaluation Metrics and Dashboard Setup.
![image alt](image url)

Findings
-------------
- **Model Performance**:
    - Initial Model (Temporal + 1,2,3 and 5-year lags):
      - MAE: ~2423.96MW, MAPE: 8.55%, R^2: 79%
    - Improved Model (Rolling 1 year mean + 1 hour, 1 day, 1 week, 1 year and 2 year lags):
      - MAE: 908.66MW, MAPE: 3.22%, R^2: 96%
- **Insights**:
    - Including both short-term (1 hour, 1 day, 1 week) and long-term (1 year, 2 year) lagged features in addition to the 1 year rolling mean feature improved model accuracy while reducing the risk of over-fitting.
    - Using a 30 minute lag (1 datapoint) increased training complexity and posed risks of overfitting.   

Interactive Dashboard
----------------------
An interactive dashboard was created using Streamlit to data and insight presentation as well as enable user interactivity by technical and non-technical stakeholders. 
To use the dashboard, navigate to the project directory and start the streamlit app using the following command in the terminal {streamlit run src/visualization/electricity_demand_dhasboard.py}
The dahsboard can be used to visual historical electricity demand and its comparison with solar and wind generation. In addition, it can be used make forecasts by inputting a forecast period or upto a maximum of 3 years and clicking generate.
