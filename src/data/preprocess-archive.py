#### Making test and training splits, pipelines for data scaling

import numpy as np
import pandas as pd

def make_timeseries_test_train_splits(df, target_col, test_size):
    """
    Creates train and test splits of data for timeseries (takes the most recent).
    
    Input
    =====
    df, dataframe, time series index.
    target_col, str, string name of column which is target variable for prediction.
    test_size, int, size of data for test set.
    
    Output
    ======
    X_train, dataframe, training set features.
    X_test, dataframe, test set features.
    y_train, series, training set labels.
    y_test, series, test set lables.

    """
    #### make splits
    X_test = df[-test_size:]
    X_train = df[:-(test_size)]

    y_test = X_test.pop(target_col)
    y_train = X_train.pop(target_col)

    #### print size of data sets
    print('DATA POINTS:')
    sizes = {'orig size': df.shape[0], 'training: ':(X_train.shape[0]), 'testing: ': X_test.shape[0]}
    for i in sizes:
        print(i,sizes[i])

    return(X_train, X_test, y_train, y_test)


def get_variable_types_lists(features_df):
    """
    Auto-assigns each feature to be either numical, catagorical, binary. Column names are assigned to a corresponding list.
    
    Input
    =====
    features_df, dataframe, containing features.
    
    Output
    ======
    num_features, list of strings, list of column names that contain numerical features.
    cat_features, list of strings, list of column names that contain catagorical features.
    bin_features, list of strings, list of column names that contain binary features.
    
    """
    
    num_features = []
    cat_features = []
    bin_features = []
    dtypes = features_df.dtypes
    
    #### find binary features    
    for feature_name, dtype in dtypes.iteritems():
        
        if dtype == np.bool: # if boolean add to binary list
            bin_features.append(feature_name)
        elif (dtype in [np.int, np.int64, np.int32, np.int16]) & (len(features_df[feature_name].unique()) == 2): # if int and only have 2 unique values then add to bin list
            bin_features.append(feature_name)
        
    #### find categorical features
    for feature_name, dtype in dtypes.iteritems():
        if (dtype in [np.int, np.int64, np.int32, np.int16]) & (len(features_df[feature_name].unique()) >= 2) & (len(features_df[feature_name].unique()) <= 12):
            cat_features.append(feature_name)
    
    #### deduce num features which remain
    for feature_name, dtype in dtypes.iteritems():
        if feature_name not in (cat_features)+(bin_features):
            num_features.append(feature_name)
    
    
    
    return(num_features, cat_features, bin_features)

def check_for_catagorical_type_difference_between_train_test(df_train, df_test, features_to_check):
    """
    Sometimes there may be issues with a new catagory being present in the test data set.
    Function checks to see if this is the case for each feature. 
    If there is a discrepancy it prints the issue and makes the feature continuous to alleaviate any future processing issue. 
    NOTE: this may result in the feature not being as useful to the model as it otherwise would.
    
    Input
    =====
    df_train, dataframe, containing training features.
    df_test, dataframe, containing testing features.
    categorical_column_list, list of strings, containing columns that are either categorical or binary features.
    
    Output
    ======
    problem_col_list, list of strings, column names for which there was a difference found between train and test sets.
    
    """
    problem_col_list = []
    
    for col_name in df_train[features_to_check].columns:
        
        train_vals = set(df_train[col_name].unique()) # make into sets so can make a comparison
        test_vals = set(df_test[col_name].unique())
        
        if train_vals != test_vals:
            print('Feature name: ', col_name)
            print('Categories in training: ', train_vals)
            print('Categories in testing: ', test_vals)
            print('')
            problem_col_list.append(col_name) # add to list to return to user

            return(problem_col_list)

def change_feature_types_to_numeric(problem_col_list, list_to_remove_from ,num_features):
    """
    Removes the columns found in problem_col_list from either catagorical or bin feature list and adds it to the numerical feature list.
    THe change is not implicit and must be assigned.
    
    Input
    =====
    problem_col_list, list of str, feature columns to be removed
    list_to_remove_from, list of str, list to be removed from
    num_features, list of str, list to be added to.
    
    Output
    ======
    list_removed_from,
    list_added_to, 
    """
    for col in problem_col_list:
        list_to_remove_from.remove(col)
        num_features.append(col)
    
    return(list_to_remove_from ,num_features)


#### a sklearn transformer class to select attributes of interest for each group
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values


class MakeBooleanAnInteger(BaseEstimator,TransformerMixin):
    def __init__(self):
        return
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X*1