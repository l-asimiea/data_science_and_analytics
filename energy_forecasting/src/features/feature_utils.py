"""
This script contains all functions relevant for featuring engineering operations
implemented on the dataset prior to or during model development. 

Input:
    df:         the dataframe containing source variables for featuring engineering
    
Output:
    data_uk:    the dataframe including created features  

"""

# Imported Libraries
import pandas as pd
import datetime
import numpy as np
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.stattools import pacf, acf
from src.visualization.visualize import plotly_user_standard_settings
plotly_user_standard_settings(pio, px)

# path to saving plots
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
save_path = os.path.join(project_root, "reports/figures/")



# ---------------------------------------------------------------------------
def create_temporal_features(df, save_data=False):
    """
    This function creates the time features; hour, day, week, month and year 
    as well as lag features lag1, 2, 3 and 5
    Args:
    df:         data as a pandas dataframe
    add_lags:   Optional switch to create lag features, default is True
    save_data:  Optional switch to save the engineering data to file, default is False
    
    Return:
    data_uk:    the feature engineered data
    """
    
    #df = pd.read_pickle(f"{project_root}/data/interim/uk_data_processed_postEDA.pkl")

    data_uk = df.copy()
    # Ensure data index is in right format
    if 'date' in data_uk.columns:
        data_uk.rename(columns={'date':'datetime'}, inplace=True)
        data_uk['datetime'] = pd.to_datetime(df['datetime'])
        data_uk.set_index('datetime', inplace=True)
    elif data_uk.index.name == 'datetime':
        if not isinstance(data_uk.index, pd.DatetimeIndex):
            try:
                data_uk.index = pd.to_datetime(data_uk.index)
            except Exception as e:
                raise ValueError(f"Failed to convert index to DatetimeIndex: {e}")
            
    # Create time_features: hour, day, week, month and/or year
    time_features = ['period_hour', 'hour','day_of_week','day_of_month', 'day_of_year',
                     'week', 'month', 'quarter','year']
    
    for col in time_features:
        if col == 'period_hour':
            data_uk['period_hour'] = (data_uk["period"]).apply(
                lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5)))
        elif col == 'hour':
            data_uk['hour'] = data_uk['period_hour'].str.split(":").str[0].astype(int)
        elif col == 'day_of_week':
            data_uk['day_of_week'] = data_uk.index.day_of_week
        elif col == 'day_of_month':
            data_uk['day_of_month'] = data_uk.index.day
        elif col == 'day_of_year':
            data_uk['day_of_year'] = data_uk.index.day_of_year
        elif col == 'week':
            data_uk['week'] = data_uk.index.isocalendar().week.astype('int64')
        elif col == 'month':
            data_uk['month'] = data_uk.index.month
        elif col == 'quarter':
            data_uk['quarter'] = data_uk.index.quarter
        elif col == 'year':
            data_uk['year'] = data_uk.index.year

    return data_uk
    
def create_lag_features(df, save_data=False):
    """
    Add lags to the dataset.
    """
        
    lags_df = df.copy()
    
    # Ensure data index is in right format
    if 'date' in lags_df.columns:
        lags_df.rename(columns={'date':'datetime'}, inplace=True)
        lags_df['datetime'] = pd.to_datetime(lags_df['datetime'])
        lags_df.set_index('datetime', inplace=True)
    elif lags_df.index.name == 'datetime' or lags_df.index.name == 'date':
        if not isinstance(lags_df.index, pd.DatetimeIndex):
            try:
                lags_df.index = pd.to_datetime(lags_df.index)
            except Exception as e:
                raise ValueError(f"Failed to convert index to DatetimeIndex: {e}")
            
    lags_df = lags_df.sort_index()
    target_map = lags_df['tsd'].to_dict()
    
    
    lags_df["lag_1hour"] = (lags_df.index - pd.Timedelta(hours=1)).map(target_map)
    lags_df["lag_1day"] = (lags_df.index - pd.Timedelta(days=1)).map(target_map)
    lags_df["lag_1week"] = (lags_df.index - pd.Timedelta(weeks=1)).map(target_map)
    lags_df["lag_1year"] = (lags_df.index - pd.Timedelta(weeks=52)).map(target_map)
    lags_df["lag_2year"] = (lags_df.index - pd.Timedelta(weeks=52*2)).map(target_map)
    lags_df['rolling_mean_1day'] = lags_df['tsd'].rolling('1D').mean()
    

    # column re-arrangement
    lags_df = lags_df[sorted(lags_df.columns)]
    
    if save_data:   
        save_path = os.path.join(project_root, "data/processed/uk_data_fe_processed.pkl")
        lags_df.to_pickle(save_path)
    
    return lags_df




def plot_correlation_matrix(df, title='Correlation Matrix', plot_save=False):
    """
    This function creates an interactive correlation matrix heatmap of the features.
    Note:
    user_standard settings function for plots should be called to maintain plot 
    consistency throughtout the project. 
    Args: 
    df:         data asa pandas dataframe 
    title:      Default title, if non is user defined
    plot_save:  Optional switch to save the plot to file, default is False
    
    Return:
    fig:        the plot        
    """
    
    correlation_matrix = round(df.corr(),2)     # round values to 2dp
    fig = px.imshow(correlation_matrix, text_auto=True,
                        title=title,
                        labels={"color":"Correlation"}
                        ).update_layout(
                            width=650, height=500
                            )
    
    if plot_save == True:
        file_name = f"{save_path}data_correlation.html"
        fig.write_html(file_name)
    return fig 


# ----------------------------------------------------------
def plot_feature_distribution(df, 
                              feature:str, 
                              target=None, 
                              bins=30, 
                              plot_save=False):
    """
    This function creates an interactive histogram for a feature, optionally colored by a target. 
    Args: 
    df:         data asa pandas dataframe 
    feature:    feature variable 
    target:     the target variable
    bins:       Default number of bins set for the histogram
    plot_save:  Optional switch to save the plot to file, default is False 
    
    Return:
    fig:        the plot
    """
    
    title = f"Distribution of {feature}"
    fig = px.histogram(
        df,
        x=feature,
        color=target,
        nbins=bins,
        marginal='box',
        title=title,
        labels={"color":"Target", "x":feature}
    ).update_layout(bargap=0.1)
    
    if plot_save == True:
        file_name = f"{save_path}feature_distribution.html"
        fig.write_html(file_name)
    return fig

def plot_seasonal_decompose(result:DecomposeResult, 
                            dates:pd.Series=None,
                            target_name:str=None,
                            plot_save=False):
    """
    This function creates a plot of a decomposed timeseries target variable in this case, TSD. 
    Args: 
    result:     result of the decomposition  
    dates:      the date series associated with the target variable
    target_name: name of the target variable
    plot_save:  Optional switch to save the plot to file, default is False 
    
    Return:
    fig:        the plot
    """
    plot_title = f"{target_name} Seasonal Decomposition" if target_name is not None else "Seasonal Decomposition"
    x_values = dates if dates is not None else np.arange(len(result.observed))
    fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"])

    fig.add_scatter(x= x_values, y=result.observed, mode='lines', row=1, col=1)
    fig.add_scatter(x= x_values, y=result.trend, mode='lines', row=2, col=1)
    fig.add_scatter(x= x_values, y=result.seasonal, mode='lines', row=3, col=1)
    fig.add_scatter(x= x_values, y=result.resid, mode='lines', row=4, col=1)
    fig.update_layout(height=900, margin={'t':100}, title_x=0.5,title=plot_title, showlegend=False)
    
    if plot_save:
        file_name = f"{save_path}seasonal_decomposition.html"
        fig.write_html(file_name)
    return fig
    


def plot_acf_and_pacf(series, lags=20):
    """
    Plots ACF and PACF for a given pandas series.
    
    Args:
    series (pd.Series): time series data
    lags (int):         Number of lags to include in the plots.
    
    Return:
    fig_acf:            ACF plot
    fig_pacf:           PACF plot
    """
    
    # Calculate ACF and PACF
    acf_values = acf(series, nlags=lags, fft=True)
    pacf_values = pacf(series, nlags=lags)
    
    # Create Dataframes for acf and pacf
    acf_df = pd.DataFrame({'Lag': range(len(acf_values)), 'ACF': acf_values})
    pacf_df = pd.DataFrame({'Lag': range(len(pacf_values)), 'PACF': pacf_values})
    
    # ACF Plot
    fig_acf = px.scatter(acf_df, x="Lag", y='ACF', 
                         title="Autocorrelation Function (ACF)")
    for lag, acf_val in zip(acf_df['Lag'], acf_df['ACF']):
        fig_acf.add_shape(
            type='line',
            x0=lag, x1=lag,
            y0=0, y1=acf_val,
            line=dict(color='black', width=2)
        )
    
    
    # PACF Plot
    fig_pacf = px.scatter(pacf_df, x="Lag", y='PACF', 
                         title="Partial Autocorrelation Function (PACF)")
    for lag, pacf_val in zip(pacf_df['Lag'], pacf_df['PACF']):
        fig_pacf.add_shape(
            type='line',
            x0=lag, x1=lag,
            y0=0, y1=pacf_val,
            line=dict(color='black', width=2)
        )
    
    
    return fig_acf, fig_pacf



