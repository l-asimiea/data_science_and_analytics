"""
Model Development 
This script contains code to train ML models using processed electricity data
in search for an adequate forecasting model. 
"""

# Imported Libraries
import numpy as np
import pandas as pd
import datetime
import plotly.express as px
import plotly.io as pio
import warnings
import os 
from src.visualization.visualize import plotly_user_standard_settings
plotly_user_standard_settings(pio, px)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from src.models.model_utils import model_evaluator, model_feature_importance, plot_actual_vs_model_pred
import pickle
import joblib

# Settings
warnings.filterwarnings('ignore')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
plot_save_path = os.path.join(project_root, "reports/figures/")

#--------------------------------------------------------------------------------  
# Load data 
data_file_path = f"{project_root}/data/processed/uk_data_fe_processed.pkl"
df = pd.read_pickle(data_file_path)
 
# Extract features and target variables 
features = [
    'lag_1day', 'lag_1hour', 'lag_1week', 'lag_1year', 'lag_2year', 'rolling_mean_1day'
]
target = 'tsd'
df = df.sort_index()
X = df[features].dropna()
y = df[target].dropna().loc[X.index]         
 
# Define models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100,random_state=42)

# Stores for model outputs 
rf_results = []
gb_results = []
xgb_results = []

#----------------------------------------------------------------------
# Create time-series split cross validation
tscv = TimeSeriesSplit(n_splits=5, test_size =48*365*1, gap=48)

# Loop through the folds...
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    print(f"Running Fold {fold}...")
    print('Spliting data into training and testing data subsets...')
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
               
    # Run Models
    try:
        # Random Forest
        print('Running Random Forest model...')
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_results.append(model_evaluator(fold, y_test, rf_pred, rf_model, 'random_forest'))  
    except Exception as e:
        print(f"Random Forest model failed on fold {fold}: {e}")
        
    try:
        # Gradient Boosting
        print('Running Gradient Boost model...')
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_results.append(model_evaluator(fold, y_test, gb_pred, gb_model, 'gradient_boost'))   
    except Exception as e:
        print(f"Gradient Boostng model failed on fold {fold}: {e}")
        
    try:
        # XGBoost
        print('Running XGBoost model...')
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_results.append(model_evaluator(fold, y_test, xgb_pred, xgb_model, 'xgboost'))
    except Exception as e:
        print(f"XGBoost model failed on fold {fold}: {e}")
          
# Model training complete
print('Model training complete')

# Best Model
best_model_vars = rf_results[0]   # best model - rf_model fold 1 
print(f"Result of the best model:{best_model_vars}")


# Feature Importance - Best Model...
print('Generating feature importance from best model')
best_model_importance, fig = model_feature_importance(X,best_model_vars)
fig.show()


"""
Note:
Feature importance plots of the best 2 models (gradient boost and random forest at 5th folds) 
showed that the lag features offered the most importance to the model training 
especially lag1
"""
#--------------------------------------------------------------------------
# Visualise Model Verification Performance
model_vars = best_model_vars
fig = plot_actual_vs_model_pred(model_vars, X, y)
file_name = f"{plot_save_path}Actual_vs_Predicted_TSD.html"
fig.show()
fig.write_html(file_name)

"""
Note: 
The actual TSD was compared to predictions made using the best model(Gradient Boost fold 5) and 
the second best model (Random Forest fold 5). 
The comparison showed that the gradient boost and random forest exibited 5.6% and 6% 
average absolute error respectively with mean absolute error of 1705.6MW and 1617MW respectively.
Visually, both models exhbited regions of mainly over prediction than under predictions
at the peaks of the actual TSD with gradient boost performing better (i.e., being closer 
to the actual TSD).
"""
#------------------------------------------------------------------
# Model Optimisation using GridSearchCV
# Define hyperparameter grid for best Gradient Boost model
""" 
# Not Done due to high computational cost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0]
}
# Perform Grid Search
best_model_grid_search = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=best_model_grid_search,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=tscv,
    verbose=2,
    n_jobs=-1)

grid_search.fit(X_train,y_train)
print("Best Gradient Boost Parameters:", grid_search.best_params_)
"""




#--------------------------------------------------------------------------
# Save best model only 
#best_model = best_model_vars['model']
joblib.dump(best_model_vars['model'], f"{project_root}/models/{best_model_vars['model_name']}_best_model.pkl")

# alternate - save model and its metrics 
model_save_filepath = '.pkl'
with open(model_save_filepath, 'wb') as file:
    pickle.dump(best_model_vars)



  


