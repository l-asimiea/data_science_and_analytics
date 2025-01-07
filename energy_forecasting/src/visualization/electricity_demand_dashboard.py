""" Create Streamlit Dashboard Code
# This code sets up a Streamlit dashboard for visualizing electricity and generation data, 
# along with a predictive model for estimating future electricity demand.
 %%writefile energy_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import joblib
import os
from datetime import datetime, timedelta
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from src.models.model_utils import model_feature_importance
from src.models.predict_model import make_forecasts
from src.features.feature_utils import plot_correlation_matrix

# Function to load  data
@st.cache_data
def load_data():
    return pd.read_pickle(f"{project_root}/data/processed/uk_data_fe_processed.pkl")

# Function to load  model
@st.cache_resource
def load_model():
    model = joblib.load(f"{project_root}/models/xgboost_2nd_best_model.pkl")
    return model

# Function to load model and model variables dictionary
@st.cache_data
def load_model_and_modelvars():
    model_save_filepath = f"{project_root}/models/xgb_models_and_vars.pkl"
    with open(model_save_filepath,'rb') as file:
        model_and_vars = pickle.load(file)
    return model_and_vars

# Function to load performance metrics
@st.cache_data
def load_performance_metrics():
    model_save_filepath = f"{project_root}/models/xgb_models_vars.pkl"
    with open(model_save_filepath,'rb') as file:
        loaded_model_vars = pickle.load(file)
    return pd.DataFrame({'model_name': loaded_model_vars['model_name'],
                         'MAE': [float(loaded_model_vars['MAE'])],
                         'RMSE': [float(loaded_model_vars['RMSE'])],
                         'MAPE': [float(loaded_model_vars['MAPE'])]
                         })
    

def main():
    st.title('Electricity Demand Forecasting Dashboard')
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    # Sidebar filters
    st.sidebar.header('Filters')
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df.index.min().date(), df.index.max().date())
    )
    forecast_days = st.sidebar.number_input("Forecast Horizon (Days)", 
                                            min_value=1, 
                                            max_value=1095,
                                            value=7,
                                            step=1)
    
    # Filter data based on date range
    mask = (df.index.date >= date_range[0]) & (df.index.date <= date_range[1])
    filtered_df = df[mask]
    
    # Historical Data Visualisation
    st.subheader('Historical Actual Electricity Demand vs. Generation')
    if not filtered_df.empty:
        fig = px.line(filtered_df, x=filtered_df.index, 
                    y=['tsd',
                        'solar_generation',
                        'wind_generation'],
                    title="Historical Electricitry: TSD, Solar and Wind Generation",
                    labels={'value': 'Power (MW)', 'datetime': 'Datetime'}).update_layout(
                        width=900
                    )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected date range.")
    
    # Correaltion Map
    st.subheader('Feature Correlation')
    corr_vars = ['tsd', 'lag_1day', 'lag_1hour', 'lag_1week', 
                 'lag_1year', 'lag_2year', 'rolling_mean_1day']
    fig = plot_correlation_matrix(df[corr_vars])
    st.plotly_chart(fig)
    
    # Electricity Demand Forecasting
    st.subheader('Electricity Forecasting')
    if st.button("Generate Forecast"):
        # Generate forecast for the prediction days
        fig = make_forecasts(model,forecast_days)
        st.plotly_chart(fig)
    else:
        st.info("Click 'Generate Forecast' to predict future values.")
    
    # Model Performance
    st.subheader('Model Performance Metrics')
    metrics_df = load_performance_metrics()
    st.dataframe(metrics_df)
    
    # Feature Importance
    st.subheader('Feature Importance')
    feat_vars = df[['lag_1day', 'lag_1hour', 'lag_1week', 
                 'lag_1year', 'lag_2year', 'rolling_mean_1day']].dropna()
    model_and_vars = load_model_and_modelvars()
    _ , fig = model_feature_importance(feat_vars,model_and_vars)
    st.plotly_chart(fig)
    
    
    
    
# Footer
#st.write("Built with dedication using Streamlit")
if __name__ == '__main__':
    main()

print("Streamlit dashboard code has been saved to 'electricity_demand_dashboard.py'")
print("To run the dashboard, use the command: streamlit run electricity_demand_dashboard.py")