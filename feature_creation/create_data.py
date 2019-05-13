import pandas as pd
import numpy as np
import itertools

def make_attendances_dataframe(num_atten,pat_per_day=10, seed=True):
    """
    creates a random df with attendances as rows of size provided by user.
    """
    #### make lists for input to df 
    atten_id = np.arange(num_atten)+1000
    num_patients = int(num_atten/pat_per_day)
    pat_id = np.random.randint(0,num_atten*1000,num_atten)
    if seed == True:
        np.random.seed(2)
    # arrivals
    num_days = int(num_atten/pat_per_day)
    start_time = pd.datetime(2018,1,1) # set a date to start the attendances at
    
    arrival_time = []
    day_counter = 0
    date = start_time
    for atten in atten_id:
        random_datetime = date + pd.Timedelta(np.random.randint(0,23),'h') + pd.Timedelta(np.random.randint(0,59),'m')
        arrival_time.append(random_datetime)
        day_counter += 1
        if day_counter == 10:
            day_counter = 0
            date = date + pd.DateOffset(1,'D')
    
        
    # watiting times
    time_in_department = np.random.randint(1,60*6,num_atten)
    
    # binary ambulance arrival
    ambulance = np.random.randint(0,2,num_atten)
    
    
    # make df
    d = {'atten_id':atten_id,
        'pat_id':pat_id,
        'arrival_datetime':arrival_time,
        'time_in_department':time_in_department,
        'ambulance_arrival':ambulance}
    df = pd.DataFrame(d)#.set_index('atten_id')

    # make departure times
    f = lambda x, y : x + pd.Timedelta(y,'m')
    df['departure_datetime'] = df.apply(lambda row: f(row['arrival_datetime'], row['time_in_department']), axis=1)
    
    # make gender
    df['gender'] = df['pat_id'].apply(lambda x : x%2)
    
    #sort values by arrival time
    df = df.sort_values('arrival_datetime')
    
    # change att dtype
    df['atten_id'] = df['atten_id'].astype('int64')
    
    return(df)


def make_timeindex_dataframe(df,col_label,freq='D',index_lab='atten_id'):
    """calculates the first and last times contained in dataframe.
    creates new df with unique hourly or daily time interval in one column."""
    df = df.set_index(index_lab) # mod for script
    # get_datetime cols
    cols = df.select_dtypes(include='datetime').columns
    start = df[cols].min().min().replace(hour=0, minute=0,second=0)#values
    end = df[cols].max().min().replace(hour=0, minute=0,second=0) +pd.Timedelta(days=1) #values
    d = pd.date_range(start,end,freq=freq)
    df_new = pd.DataFrame({col_label:d})
    
    return(df_new)

# make active_attendaces - use code from previosu script

def make_HourlyTimeAttenNum_dataframe(df,arrival_col,departure_col):
    """
    inputs:
    df with attendance number as index,
    arrival, departure datetime col names (must be datetime format)
    ouptut:
    df, contains many-to-many link between the arrival_
    """
    df1 = df[[arrival_col,departure_col]].copy()
    df1[arrival_col] = df1[arrival_col].apply(lambda x : x.replace(second=0,minute=0)) # round arrival hour down
    df1[departure_col] = df1[departure_col].apply(lambda x : x.replace(second=0,minute=0)) +pd.Timedelta(hours=1) # round leaving tim up
    
    #### create col with number of hours active 
    df1['n_hours'] = ((df1[departure_col] - df1[arrival_col])/pd.Timedelta(1,'h')).astype(int)
    df1 = df1.drop(df1[df1['n_hours'] <=0].index) # must drop the negative n_hour rows as otherwise messes up my size of array initilisation (was getting an error that the index i was assinging to in ids was out of bounds).
    
    #### time efficient (i hope) function for cycling through and finding all combinations of active hours for attednaces - create a (long format) list of links between attendance numbers and 

    # function for list comp which finds list of datetimes (for each hour)
    date_func = lambda datetime , offset : datetime + pd.Timedelta(offset,'h')

    # iterate over rows in df
    df1 = df1.reset_index() # reset so have the new index to itereate over

    ids = np.empty(shape=(df1['n_hours'].sum()),dtype='int64') # initilise array - change to np.empty() to speed up
    timestamps = np.empty(shape=(df1['n_hours'].sum()),dtype='datetime64[s]')
    row_count = 0

    for row in df1.itertuples():
        atten_id = [row[1]]
        hour_list = [date_func(row[2],i) for i in np.arange(row[4])] # creates list of hour datetimes

        # create array of list for all combinations of timestamp
        for i in itertools.product(atten_id,hour_list):
            ids[row_count] = i[0] # assign patient numbers
            timestamps[row_count] = i[1]
            row_count += 1 # add to row count for new array    
    # put into df
    data = {'atten_id':ids,
       'hour':timestamps}
    df_new = pd.DataFrame(data=data)

    return(df_new)
