#### Making test and training splits, pipelines for data scaling

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


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



class DataFrameSelector(BaseEstimator,TransformerMixin):
    """ Selects only columns provided as list when initialised.
    """
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names]#.values

class DataFrameReform(BaseEstimator,TransformerMixin):
    " Takes numpy array and forms into dataframe with column names."
    def __init__(self, cols_to_include):
        self.cols_to_include = cols_to_include
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return (pd.DataFrame(X,columns=self.cols_to_include))

class MakeBooleanAnInteger(BaseEstimator,TransformerMixin):
    def __init__(self):
        return
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X*1