# Main function 
def read_and_proc_csvdata(file_path):
    
    import pandas as pd
    import os
    
    """ 
    This function reads and processes the electricity data and processes it into a
    a form useful for analysis up to and before the feature engineering stage of 
    the project.
    
    Inputs: 
        file_path:      path the the csv file location
    
    Outputs:
        cln_df:         processed data output and saved in pickle format in
                        energy_forecasting/data/interim 
    """
    
    # 1. Read data file 
    if len(file_path) == 0:
        Warning("file_path not provided. In-built file path used")
        file_path = "../../data/raw/uk_electricity_consumption_historic_demand_2009_2024.csv"
    else:
        file_path = file_path  # Use input file_path
    df = pd.read_csv(file_path)
    
    # 2. Select only the columns of interest 
    focus_columns = [
        cols for cols in df.columns
        if (cols.startswith('settlement_') 
        or cols.startswith('nd')
        or cols.startswith('tsd')
        or cols.startswith('england_')
        or cols.startswith('embedded_')
        or cols.endswith('_holiday'))
    ]
    
    cln_df = df[focus_columns]

    
    # 3. Remove unwanted prefixes and suffixes in column names
    cln_df_columns = [
        column_name.replace('embedded_','').strip()
        .replace('settlement_','').strip()
        for column_name in cln_df.columns 
    ]
    
    cln_df.columns = cln_df_columns
    
    # 4 Change date format
    cln_df['date'] = pd.DatetimeIndex(cln_df['date'])
    
    # 5. Isolate load & generation data columns for further processing
    demand_n_gen_cols = [col for col in cln_df.columns.tolist()
                       if col == 'nd' 
                       or col == 'tsd'
                       or col.endswith('_demand')
                       or '_generation' in col]
    # Resolve Nan values
    # 4.1. Interpolate continuous time-series data columns (demand and generation) linearly as there are few missing values 
    # where data is expected to follow a linear or seasonal trend
    #cln_df[demand_n_gen_cols] = cln_df[demand_n_gen_cols].interpolate(method='linear')

    # 4.2. Fill missing capacity columns with median values as they are non-time-senstive and exhibit stability
    #cln_df['solar_capacity'].fillna(cln_df['solar_capacity'].median(), inplace=True)
    #cln_df['wind_capacity'].fillna(cln_df['wind_capacity'].median(), inplace=True)

    # 4.3. Impute profile values based on average by hour because profile data are time-sensitive
    #cln_df['hour'] = cln_df['utc_timestamp'].dt.hour
    #cln_df['solar_profile'] = cln_df.groupby('hour')['solar_profile'].transform(lambda x: x.fillna(x.mean()))
    #cln_df['wind_profile'] = cln_df.groupby('hour')['wind_profile'].transform(lambda x: x.fillna(x.mean()))

    # 4.4. Forward and backward fill for any remaining NaNs
    #cln_df.fillna(method='ffill', inplace=True)
    #cln_df.fillna(method='bfill', inplace=True)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    save_path = os.path.join(project_root, "data/interim/uk_data_interim_proc_01.pkl")
    
    cln_df.to_pickle(save_path)
    
    return cln_df