# -*- coding: utf-8 -*-

# Credit to Scollay Petry at 
# (https://python.plainenglish.io/storing-pandas-98-faster-disk-reads-and-72-less-space-208e2e2be8bb)
# GitHub repository: https://github.com/scollay/caffeinated-pandas

# Imports
import numpy as np
import os
import pandas as pd

# Local imports
import caffeinated_pandas_utils as cp

def write_Parquet_data(experiment_dir: str, 
                       file_dir_and_name: str, 
                       time_series: list
                      ) -> None:
    """
    Info: 
        This function takes a time series array and writes the data to a Parquet
        file with brotli compression.
    
    Input:
        experiment_dir: the directory to which the results are saved.
        
        file_dir_and_name: a string containing the directory and filename of the
            data to be saved to.
        
        time_series: a 1D (or 2D) array containing one (or more) asset price 
            time series.
        
    Output:
        A Parquet file containing the input time series data with brotli compression
        saved in the local data directory.
    
    """
    
    # Initialize directories and if non-existent create them
    os.chdir(os.getcwd())
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results\\')
    data_dir = os.path.join(experiment_dir, 'Data\\')
    
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
        
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        
    # Create pandas DataFrame object of time_series array
    df = pd.DataFrame(time_series)
    
    if len(df.columns) == 1:
        df.columns = ['1']
    elif len(df.columns) > 1:
        df.columns = [f'Path {idx+1}' for idx in range(time_series.shape[1])]
    cp.write_file(df=df, fn=file_dir_and_name, compression='')
    
    # Delete DataFrame to clear memory
    del df
    
def read_Parquet_data(file_dir_and_name: str
                     ) -> list:
    """
    Info:
        This function reads in data from a Parquet file with brotli compression 
        and returns the contained data in a numpy array.
    
    Input:
        file_dir_and_name: a string containing the directory and filename of the
            data Parquet file with brotli compression to be read from.
    
    Output:
        A numpy array containing the read-in data.
    
    """
    # Read data file to pandas DataFrame and convert to numpy array
    df = cp.read_file(file_dir_and_name, compression='')
    time_series = df.to_numpy()
    
    # Delete DataFrame to clear memory
    del df
    
    return time_series

def convert_Parquet_to_CSV(file_dir_and_name):
    """
    Info:
        This function reads in data from a Parquet file with brotli compression 
        and exports the data in a CSV file.
    
    Input:
        file_dir_and_name: a string containing the directory and filename of the
            data Parquet file with brotli compression to be read from.
    
    Output:
        A CSV file containing the read-in data.
    
    """
    # Read data file to pandas DataFrame and convert to numpy array
    df = cp.read_file(file_dir_and_name, compression='')
    time_series = df.to_numpy()
    
    # Save numpy array as CSV file
    filename = file_dir_and_name.split("\\")[-1]
    new_file_dir_and_name = os.path.splitext(file_dir_and_name)[0] + '.csv'
    np.savetxt(new_file_dir_and_name, time_series, delimiter=",")
    print(f'\n{filename} was converted to a CSV file and saved to {new_file_dir_and_name}')
    
    # Delete DataFrame to clear memory
    del df    