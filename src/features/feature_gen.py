

def create_lagged_vars_array(df, col_name, number_lags):
    ""
    for i in np.arange(1, number_lags + 1):
        df[col_name+'_lag'+str(i)] = df[col_name].shift(i)
    return(df)

def create_lagged_vars_list(df, col_name, lags_list):
    """
    Takes a df and the column indicated in col_name, creates new columns in the df with lagged variables. The Lags correpsond to those which are given in the lasg_list.
    Parameter
    =========
    df, dataframe,
    col_name, str, name of column to perform lags on.
    lags_list, list of int, lags at which to perform lagging.
    
    Return
    =======
    df, dataframe, with new columns. NOTE: although returned the changes will already be made to original df.
    """
    for i in lags_list:
        df[col_name+'_lag'+str(i)] = df[col_name].shift(i)
    return(df)

def make_col_derivative(df, column_name ,deriv_lag, num_diffs):
    """
    Parameters
    ==========
    df, dataframe, 
    column_name, str, name of column
    deriv_lag, int, number of points to lag over
    
    Returns
    =======
    derivative, Series, new derivative column
    """
    new_col_name = column_name + '_deriv' + str(deriv_lag)
    
    #### Make time column in new df
    deriv_df = pd.DataFrame(np.arange(0, len(df)),columns = ['time'], index = df.index)
    
    ####
    deriv_df['X'] = df[column_name]
    crude_deriv = deriv_df['X']
    for i in np.arange(1,num_diffs+1):
        crude_deriv = (crude_deriv.diff(deriv_lag)) #/(deriv_df['time'].diff(deriv_lag))
    deriv_df[new_col_name] = crude_deriv
    deriv_df.bfill(inplace=True)
    
    
    derivative = deriv_df[new_col_name]

    return(derivative)

def perform_deriv_cals_multiple_columns(df, columns_dict, num_diffs):
    """
    Calls make_col_derivatives on multiple columns based on user input of dict.
    Parameter
    =========
    df, dataframe,
    columns_dict, dict, columns (str): list (int) pairs. Column names and lags to perform over.
    num_diffs, int, number of differentials to perform (max 2).

    Return
    ======
    df, dataframe, NOTE: new columns will aready have been assigned to dataframe without needing this assingment.
    
    """
    #### loop over each column
    for column in columns_dict.keys():
        #### loop over each lag requested
        for deriv_lag in columns_dict[column]:
            
            if num_diffs >= 1:
                new_col_name = column + '_deriv1' + '_lag' + str(deriv_lag) # get new name for column
                df[new_col_name] = make_col_derivative(df, column, deriv_lag, 1)
            if num_diffs >= 2:
                new_col_name = column + '_deriv2' +'_lag' + str(deriv_lag)
                df[new_col_name] = make_col_derivative(df, column, deriv_lag, 2)
    return(df)