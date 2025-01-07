# Imported Libraries
import pandas as pd
import datetime
import warnings
import os
import plotly.express as px
import plotly.io as pio
from src.visualization.visualize import plotly_user_standard_settings
plotly_user_standard_settings(pio, px)
from src.features.feature_utils import create_lag_features

# Settings
warnings.filterwarnings('ignore')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
plot_save_path = os.path.join(project_root, "reports/figures/")

# Forecast function
def make_forecasts(trained_model, prediction_days:int, plot_save=False):
    """ 
    This function uses a provided trained model and number of 
    prediction days to create electricity demand forecasts based on cleaned 
    and processed historical Transmission System Demand (TSD) data. 
    
    Args: 
    trained_model:      the model used for forecast creation
    prediction_days:    the number of days to forecast
    plot_save:          option to save generated plot
    
    Return:
    fig:                plot of the historic and forecasted electricty demand
    """
    
    # Load data used for model training as primary dataset
    data_file_path = f"{project_root}/data/processed/uk_data_fe_processed.pkl"
    print(f"Loading data from {data_file_path}")
    df = pd.read_pickle(data_file_path) 
    
    df = df[['lag_1day', 'lag_1hour', 'lag_1week', 
             'lag_1year', 'lag_2year', 'rolling_mean_1day','tsd']].copy()
    
    # Fit trained model on all of the primary dataset
    print("Using features from the model training stage...")
    features = ['lag_1day', 'lag_1hour', 'lag_1week', 
                'lag_1year', 'lag_2year', 'rolling_mean_1day']
    target = 'tsd'
    X_Train = df[features].dropna()
    y_Train = df[target].dropna().loc[X_Train.index]
    trained_model.fit(X_Train, y_Train)
    
    print("Creating future space for forecasting...")
    df_last_datetime = df.index.max()
    next_datetime = df_last_datetime + pd.Timedelta(minutes=30)
    if not isinstance(prediction_days, int):
        raise TypeError("prediction_days must be an integer")
    else:
        future = pd.date_range(
            start=next_datetime,
            end=next_datetime + datetime.timedelta(days = prediction_days),
            freq = "30min"
            )
    future_df = pd.DataFrame(index=future)
    
    
    # This is only needed if temporal features are present in df - commented when not needed
    #datapoints_per_day = 48
    #df_last_period = df['period'].iloc[-1]
    #future_datasteps = len(future_df) #datapoints_per_day * prediction_days
    #future_df['period'] = [(df_last_period + i - 1)%datapoints_per_day+1 for i in range(1, (future_datasteps)+1)]
    
    
    future_df['isFuture'] = True
    df['isFuture'] = False
    
    # Combined df and future_df
    past_and_future_df = pd.concat([df, future_df])
    
    # Reprocess time and lag features
    print("Reprocessing time and lag features for past and forecast timeframe...")
    
    # build to populate missing values in the future section and avoid clipping
    target_map = past_and_future_df['tsd'].to_dict()
    past_and_future_df['control_lag'] = (past_and_future_df.index - pd.Timedelta(days=prediction_days)).map(target_map)
    past_and_future_df['control_lag'] = past_and_future_df['control_lag'].fillna(method='ffill')
    
    missing_indices = past_and_future_df[past_and_future_df['tsd'].isna()].index
    past_and_future_df.loc[missing_indices, 'tsd'] = past_and_future_df['control_lag'].loc[missing_indices]
    
    
    past_and_future_df = create_lag_features(past_and_future_df)
    #past_and_future_df.drop('is_holiday', axis=1, inplace=True)
    past_and_future_df = past_and_future_df.query('isFuture').copy()
    
    # Make Predictions
    past_and_future_df = past_and_future_df[features].dropna()
    print("Forecasting over prediction days...")
    past_and_future_df['prediction'] = trained_model.predict(past_and_future_df)
    
    # Plot past and future TSD electircity 
    past_timeframe = df.index
    past_and_future_df.sort_index()
    forecast_timeframe = past_and_future_df.index
    
    past_and_forecasted_df = pd.DataFrame(index=past_timeframe.append(forecast_timeframe))
    past_and_forecasted_df['tsd'] = pd.concat([df['tsd'], past_and_future_df['prediction']])
    past_and_forecasted_df['type'] = ['Actual']*len(past_timeframe) + ['Predicted']*len(forecast_timeframe)

    fig = px.line(past_and_forecasted_df,
                  x=past_and_forecasted_df.index,
                  y='tsd',
                  color='type',
                  title="Actual vs Predicted TSD",
    ).update_layout(
        height=600,
        xaxis_title='DateTime',
        yaxis_title='Demand (MW)',
        legend_title='Demand Type'
    )
    print("Time series plot of past and forecasted TSD created")
    if plot_save == True:
        file_name = f"{plot_save_path}Actual_vs_Predicted_TSD.html"
        fig.write_html(file_name)
        print("Time series plot of past and forecasted TSD saved")
    
    return fig




"""
# Using all data for training
X_all = df[features].dropna()
y_all = df[target].dropna().loc[X_all.index]
model_for_future = best_model
model_for_future.fit(X_all, y_all)
prediction_days = 364       # number of future days to predict over
future = pd.date_range(
    str(df.index.max())[0:10],
    df.index.max() + datetime.timedelta(days=prediction_days),
    freq="30min",
)
future_df = pd.DataFrame(index=future)
last_df_period = df['period'].iloc[-1]
future_df['period'] = [(last_df_period + i - 1)%48+1 for i in range(1, (48*365)+1)]
future_df['isFuture'] = True
df['isFuture'] = False

# Create a dataframe containing the original data and the predict df
df_and_future = pd.concat([df, future_df])
from src.features.build_features import create_time_and_lag_features
df_and_future = create_time_and_lag_features(df_and_future)

# Drop holiday column as it was of very low importance contribution in the model
df_and_future.drop('is_holiday', axis=1, inplace=True)
future_df_w_features = df_and_future.query('isFuture').copy()

# Predict
future_df_w_features_for_pred = future_df_w_features[features].dropna()
future_df_w_features_for_pred['prediction'] = model_for_future.predict(future_df_w_features_for_pred)

# Plots avialable and future TSD
df_time = df.index
future_df_w_features_for_pred.sort_index()
prediction_time = future_df_w_features_for_pred.index
extended_time = df_time.append(prediction_time)

combined_df = pd.DataFrame(index= extended_time)
combined_df['tsd'] = pd.concat([df['tsd'], future_df_w_features_for_pred['prediction']])
combined_df['type'] = ['Actual']*len(df_time) + ['Predicted']*len(prediction_time)

fig = px.line(combined_df, 
              x = combined_df.index,
              y = 'tsd',
              color = 'type', 
              title="Actual vs Predicted TSD",
              ).update_layout(
                  xaxis_title='DateTime',
                  yaxis_title='Demand (MW)',
                  legend_title='Demand Type'
              )


file_name = f"{plot_save_path}Actual_vs_Predicted_TSD.html"
fig.show()
fig.write_html(file_name)
"""