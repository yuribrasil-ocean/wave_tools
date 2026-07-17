#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:47:33 2025

# Script: NDBC_and_WAVERYS_data_statistics.py

# Author: Yuri Brasil

# e-mail: yuri.brasil@oceanica.ufrj.br

# Modification: May 8, 2025

# Objective: Create a table with all statistical parameters from
             wave_statistical_parameters library.

"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import wave_statistical_parameters as wv_stats

# Getting the start time of the script execution
start_time = dt.datetime.now()

##################### Reading csv files as DataFrames #########################

# Current path
my_path = os.getcwd()

# Data path
data_path = os.path.join(my_path,'data/')

# NDBC 51001
ndbc_51001_df = pd.read_csv(f'{data_path}ndbc_51001_df_final.csv', index_col=[0]) 

# NDBC 51001
ndbc_51101_df = pd.read_csv(f'{data_path}ndbc_51101_df_final.csv', index_col=[0]) 

# WAVERYS
waverys_hawaii_df = pd.read_csv(f'{data_path}waverys_hawaii_point_df.csv', index_col=[0]) 

# loop to replace values
for df in [ndbc_51001_df, 
           ndbc_51101_df, 
           waverys_hawaii_df]:
    
    # Replacing -9999 by np.NaN
    df['Hs'] = np.where(df['Hs'] == -9999, np.NaN, df['Hs'])
    df['Tp'] = np.where(df['Tp'] == -9999, np.NaN, df['Tp'])
    df['Dp'] = np.where(df['Dp'] == -9999, np.NaN, df['Dp'])

    # my_dfs_timeseries.append(df)

# Find rows where all values are NaN in either NDBC DataFrame
nan_rows_51001 = ndbc_51001_df.isna().all(axis=1)
nan_rows_51101 = ndbc_51101_df.isna().all(axis=1)

# Combine the masks: remove rows where any of the two has all NaNs
nan_rows_combined = nan_rows_51001 | nan_rows_51101

# Apply the mask to filter out those rows in all three DataFrames
ndbc_51001_df = ndbc_51001_df[~nan_rows_combined].reset_index(drop=True)
ndbc_51101_df = ndbc_51101_df[~nan_rows_combined].reset_index(drop=True)
waverys_hawaii_df = waverys_hawaii_df[~nan_rows_combined].reset_index(drop=True)

hs_ndbc_51001 = ndbc_51001_df['Hs']
hs_ndbc_51101 = ndbc_51101_df['Hs']
hs_waverys_hawaii = waverys_hawaii_df['Hs']

list_of_hs = [hs_ndbc_51001, hs_ndbc_51101, hs_waverys_hawaii] 

# Names for identification
names = ['51001', '51101', 'WAVERYS']

# Pairwise comparisons: (name1, series1), (name2, series2)
comparisons = []

for i, (name1, series1) in enumerate(zip(names, list_of_hs)):
    for j, (name2, series2) in enumerate(zip(names, list_of_hs)):
        comparisons.append(((name1, name2), (series1, series2)))

# List to collect all stats dictionaries
all_stats = []

# Loop through comparisons
for (name1, name2), (x, y) in comparisons:
    slope, intercept, rvalue, _, _ = wv_stats.r_coefficient(x, y)
    
    stats = {
        'Reference': name1,
        'Comparison': name2,
        'BIAS': np.round(wv_stats.bias(x, y),4),
        'NBIAS': np.round(wv_stats.norm_bias(x, y),4),
        'MAE': np.round(wv_stats.mae(x, y),4),
        'NMAE': np.round(wv_stats.norm_mae(x, y),4),
        'RMSE': np.round(wv_stats.rmse(x, y),4),
        'NRMSE': np.round(wv_stats.norm_rmse(x, y),4),
        'SI': np.round(wv_stats.scatter_index(x, y),4),
        'SI_Mentaschi': np.round(wv_stats.ment_scatter_index(x, y),4),
        'FOEX': np.round(wv_stats.foex(x, y),4),
        'r': np.round(rvalue,4),
        'R²': np.round(wv_stats.r2_coefficient(x, y),4),
        'Regression Line': f'{np.round(slope,4)}x + {np.round(intercept,4)}',
        'N': len(x)
    }
    
    all_stats.append(stats)

# Create DataFrame from all stats
results_df = pd.DataFrame(all_stats)

# Save to CSV
output_file = os.path.join(my_path, 'wave_statistics_comparison.csv')
results_df.to_csv(output_file, index=False)

print(f'Statistics saved to: {output_file}')

# Getting the end time and printing running time
end_time = dt.datetime.now()
print('Duration: {}'.format(end_time - start_time))


