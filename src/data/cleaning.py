#### File with all cleaning utilities

def remove_columns_with_single_value(df):
    """ 
    Looks for columns in df which contain only one value, removes them from df.
    
    Input
    =====
    df, dataframe, 
    
    Output
    =======
    df, dataframe, with dropped columns
    """
    #### searhc for columns
    cols_to_drop = []
    for col in df.columns:
        col_values_list = df[col].unique()
        if len(col_values_list) == 1:
            cols_to_drop.append(col) # if column not unique add to list
    
    print("Columns being dropped which contain single values: ", cols_to_drop)
    df = df.drop(cols_to_drop, axis=1)

    return(df)


def find_duplicate_column_names(frame):
    "finds columns with duplicate values in each row. Returns list of strings."
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            iv = vs.iloc[:,i].tolist()
            for j in range(i+1, lcs):
                jv = vs.iloc[:,j].tolist()
                if iv == jv:
                    dups.append(cs[i])
                    break

    return dups

def remove_duplicate_columns(df):
    "Calls find_duplciate_column_names and removes the columns it has found."
    find_duplicate_column_names(df)
    return()

    
def tidy_column_heads(df):
    """
    Function makes all column names lowercase and replaces any whitespace with _'s. Removes wild characters.
    Input: df, df, df for cleaning
    Return: df, df, dataframe with tidy names
    """
    rename_cols = dict() # make dictionary of old column names and new ones
    for i in df.columns:
        j = i.lower()
        j = j.strip()
        j = j.replace(' ','_')
        j = j.replace('.','_')
        j = j.replace('?','_')
        j = j.replace('&','_')
        j = j.replace('%','perc')
        j = j.replace('(','')
        j = j.replace(')','')
        j = j.replace(':','')
        j = j.replace(';','')
        j = j.replace('-','')
        j = j.replace('/','')
        #j = j.replace('\','')
        rename_cols[i] = j

    df = df.rename(columns=rename_cols)
    return(df)
