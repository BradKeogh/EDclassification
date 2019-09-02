# io file contains functionality to load data

import pandas as pd
import numpy as np

def import_and_prep_target_df_from_pickle(path_to_pickle, target_col_name = 'EDMeanOcc'):
    """
    Inputs:
    ------
    path_to_pickle, str, full path to pickle of pandas dataframe.
    target_col_name, str, name of column in dataframe which contains the continuous value used to calc flag_target class col.
    
    Return
    ------
    df, padnas df, target class dataframe.
    """
    
    df = pd.read_pickle(path_to_pickle)
    
    #### shift EDMeanOcc so wont be a feature in models.
    df['EDMeanOcc_prevday'] = df.EDMeanOcc.shift(1)
    df.drop(columns = 'EDMeanOcc', inplace=True)
    
    #### make lagged feature for prev day class
    # df['flag_target_prevday'] = df.flag_target.shift(1) # removed as unsure as to the usability of this in production at present.
    
    return(df)

def import_and_merge_feature_dfs_from_pickles(path_to_data_folder, pickle_names, class_dataframe):
    """
    Imports all pickled files in pickle_names. 
    Pickle files must all be in folder named in path_to_data.
    All dataframes must have a datetime index.
    Merge with 'left outer join' so should have index that matches the class_dataframe that start with. 

    Inputs
    ------
    path_to_data_folder, string, path must be given in style: './../../data/'
    pickle_names, list of strings, each item must have full filename including extension i.e. .pkl
    class_dataframe, pandas df, containing 

    """
    feature_dfs = [class_dataframe]

    #### for each item in pickle_names create path and then import pickle as df
    for name in pickle_names:
        import_path = path_to_data_folder + name # make filename
        dft = pd.read_pickle(import_path)
        # append new imported df into feature_dfs list
        feature_dfs.append(dft)

    merged_dfs = merge_dfs_with_time_index(feature_dfs)

    return(merged_dfs)





def merge_dfs_with_time_index(dfs_list):
    """
    Takes list of dataframes and merges each according to time index
    """

    fm = dfs_list[0]
    for df in dfs_list[1:]:
        fm = fm.merge(df, right_index=True, left_index=True, how = 'left')

    return(fm)


