# Imported Libraries
"""
This script  holds a colletion of functions used for ML modelling 
activities in the project
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.express as px
import plotly.io as pio
from src.visualization.visualize import plotly_user_standard_settings

# Settings
plotly_user_standard_settings(pio, px)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def model_evaluator(fold, y_test, y_pred, model, model_name:str):
    """
    This function takes in model training information involving
    cross validation (cv) splits and calculates model metrics.
    
    Args:
    fold:       the fold count from the cv split
    y_test:     portion of the target variable used for testing
    y_pred:     portion of the target varibale used for prediction
    model:      the trained model
    model_name: the name of the trained modelß
    
    Return:
    {model_name, model, fold, RMSE, MAPE, R2}
    """
    mae = mean_absolute_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return ({'model_name':model_name,
             'model':model,
                'fold':fold,
                'MAE':round(mae,2),
                'RMSE':round(rmse,2),
                'MAPE': round(mape,2),
                'R2':round(r2,2)}
            )


def model_feature_importance(X,model_vars, save_plot=False):
    """
    This function uses the features (X) and model evaluator output of 
    a train a model (see model_evaluator function) 
    to generate model feature importance values and plot.
    
    Args:
    X:          features of pandas dataframe for model training 
    model_vars: dictionary containing model_evaluator function output
    save_plot:  option switch to save generated plot
    
    Return:
    importance: generate model features importance
    fig:        plot 
    """
    importance = pd.DataFrame({
        'features': X.columns,
        'importance': model_vars['model'].feature_importances_
        }).sort_values('importance', ascending=False)
    
    # Feature Importance Plot
    fig = px.bar(importance, x='importance', y='features', orientation='h',
             title=f"Feature Importance from {model_vars['model_name']} Fold {model_vars['fold']}"
             ).update_layout(width=500, height=600, margin=dict(l=120))
    
    if save_plot:
        plot_save_path = os.path.join(project_root, "reports/figures/")
        file_name = f"{plot_save_path}Actual_vs_Predicted_TSD.html"
        fig.write_html(file_name)
        
        
    return importance, fig

def plot_actual_vs_model_pred(model_vars, X, y):
    """
    This function plots a comparison on the actual and predicted data for model verification.
     
    Args:
    model_vars:         a dictionary containing the variables of the best trained model
    X:                  the features that was used for training the best model
    y:                  the target that was used for training the best model
    Return:
    fig:                the plotly plot output
    
    """
    train_set = 0
    test_set = 1
    tscv = TimeSeriesSplit(n_splits=5, test_size =48*365*1, gap=48)
    
    
    fold_ct = model_vars['fold'] - 1 # 0, 1, 2, 3, or 4 representing fold 1, 2, 3, 4 or 5 respectively 
    scope_test_idx = list(tscv.split(X))[fold_ct][test_set]
    X_scope_test = X.iloc[scope_test_idx]
    y_scope_test = y.iloc[scope_test_idx]
    
    model_pred = model_vars['model'].predict(X_scope_test)
    fig = px.line(title=f"Actual vs Predicted TSD from fold {model_vars['fold']}")
    fig.add_scatter(y=y_scope_test.values, mode='lines', name='Actual')
    fig.add_scatter(y=model_pred, mode='lines', name=f"{model_vars['model_name']} Predicted")
    fig.update_layout(yaxis_title='Transmission Systems Demand (MW)')
    
    return fig