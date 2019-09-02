# io file contains functionality to load data

import pandas as pd
import numpy as np

def import_pickled_feature_dfs(path_to_data, pickle_names):
    """
    Imports all pickled files in pickle_names. Pickle files must all be in folder named in path_to_data.

    Inputs
    ------
    path_to_data, string, path must be given in style: './../../data/'
    pickle_names, list of strings, each item must have full filename including extension i.e. .pkl 

    """
    feature_dfs = []

    #### for each item in pickle_names create path and then import pickle as df
    for name in pickle_names:
        import_path = path_to_data + name # make filename
        df = pd.read_pickle(import_path)
        # append new imported df into feature_dfs list
        feature_dfs.append(df)

    merged_dfs = merge_dfs_with_time_index(feature_dfs)

    return(merged_dfs)

def import_merge_pickled_target_class(path_to_data, pickle_name, features_df):
    """
    Import pickled target dataframe and merge with feature df (based on time index).

    Inputs
    ======
    path_to_data, string, path must be given in style: './../../data/'
    pickle_names, list of strings, each item must have full filename including extension i.e. .pkl
    features_df, df, containing all the features

    Returns
    =======
    df, df, merged target-features.

    """
    full_path = path_to_data + pickle_name
    target_df = pd.read_pickle(full_path)

    #### add target to features df
    df = pd.DataFrame(target_df['flag_target']).merge(features_df, right_index=True, left_index=True)


    return(df)


def import_merge_prevday_target_column(path_to_data, pickle_name, features_df):
    """
    Import pickled target dataframe and merge with feature df (based on time index).

    Inputs
    ======
    path_to_data, string, path must be given in style: './../../data/'
    pickle_names, list of strings, each item must have full filename including extension i.e. .pkl
    features_df, df, containing all the features

    Returns
    =======
    df, df, merged target-features.

    """
    full_path = path_to_data + pickle_name
    target_df = pd.read_pickle(full_path)

    #### add target to features df
    df = pd.DataFrame(target_df['flag_target']).merge(features_df, right_index=True, left_index=True)

    #### add previous day occupancy (NOTE: unsure when using this from legacy pickle if this is a max or a mean)
    EDocc_MAX_prevday = pd.DataFrame(target_df['EDocc']).rename(columns={'EDocc':'EDoccMAX_prevday'})

    EDocc_MAX_prevday.index = EDocc_MAX_prevday.index.shift(1,'d') # this shifts to previous day values

    df = EDocc_MAX_prevday.merge(features_df,right_index=True, left_index=True)


    return(df)



def merge_dfs_with_time_index(dfs_list):
    """
    Takes list of dataframes and merges each according to time index
    """

    fm = dfs_list[0]
    for df in dfs_list[1:]:
        fm = fm.merge(df, right_index=True, left_index=True)

    return(fm)


