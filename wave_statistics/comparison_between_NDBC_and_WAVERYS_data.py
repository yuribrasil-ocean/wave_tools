#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:06:59 2025

# Script: comparison_between_NDBC_and_WAVERYS_data.py

# Author: Yuri Brasil

# e-mail: yuri.brasil@oceanica.ufrj.br

# Modification: May 7, 2025

# Objective: Plot a timeseries of Hs, Tp and Dp from PNBOIA buoys to be used 
             for validation

"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from datetime import datetime
from datetime import timedelta
import wave_data_comparison_plots as wv_val
import wave_statistical_parameters as wv_stats

##################### Reading csv files as DataFrames #########################

# Current path
my_path = os.getcwd()

# Data path
data_path = os.path.join(my_path,'data/')

# NDBC 51001
ndbc_51001_df_timeseries = pd.read_csv(f'{data_path}ndbc_51001_df_final.csv', index_col=[0]) 

# NDBC 51001
ndbc_51101_df_timeseries = pd.read_csv(f'{data_path}ndbc_51101_df_final.csv', index_col=[0]) 

# WAVERYS
waverys_hawaii_df_timeseries = pd.read_csv(f'{data_path}waverys_hawaii_point_df.csv', index_col=[0]) 

# loop to replace values
for df in [ndbc_51001_df_timeseries, 
           ndbc_51101_df_timeseries, 
           waverys_hawaii_df_timeseries]:
    
    # Replacing -9999 by np.NaN
    df['Hs'] = np.where(df['Hs'] == -9999, np.NaN, df['Hs'])
    df['Tp'] = np.where(df['Tp'] == -9999, np.NaN, df['Tp'])
    df['Dp'] = np.where(df['Dp'] == -9999, np.NaN, df['Dp'])

    # my_dfs_timeseries.append(df)

# Find rows where all values are NaN in either NDBC DataFrame
nan_rows_51001 = ndbc_51001_df_timeseries.isna().all(axis=1)
nan_rows_51101 = ndbc_51101_df_timeseries.isna().all(axis=1)

# Combine the masks: remove rows where any of the two has all NaNs
nan_rows_combined = nan_rows_51001 | nan_rows_51101

# Apply the mask to filter out those rows in all three DataFrames
ndbc_51001_df = ndbc_51001_df_timeseries.copy()[~nan_rows_combined].reset_index(drop=True)
ndbc_51101_df = ndbc_51101_df_timeseries.copy()[~nan_rows_combined].reset_index(drop=True)
waverys_hawaii_df = waverys_hawaii_df_timeseries.copy()[~nan_rows_combined].reset_index(drop=True)

# Data title list
buoys_str = ['ndbc_51001','ndbc_51101','waverys_hawaii']

# Dictionary of timeseries
dict_dfs_timeseries = {buoys_str[0]:ndbc_51001_df_timeseries,
                       buoys_str[1]:ndbc_51101_df_timeseries,
                       buoys_str[2]:waverys_hawaii_df_timeseries}

# Dictionary of data
dict_dfs = {buoys_str[0]:ndbc_51001_df,
            buoys_str[1]:ndbc_51101_df,
            buoys_str[2]:waverys_hawaii_df}

# Create the peak frequency column
for key in buoys_str:
    dict_dfs[key]['Fp'] = 1/dict_dfs[key]['Tp']

#################### Preparing the datetime strings ###########################

# date_strings = ndbc_51001_df.index.dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist()

# my_dates = pd.to_datetime(ndbc_51001_df_timeseries.index)
# datetime_list = my_dates.to_list()

# day_month_list = []

# for d in datetime_list:
    
#     # my_datetime = datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
    
#     day_month_list.append(d.strftime('%d/%m' ))


######################### Select the time interval ############################

# Time intervals

five_months = 5*8*30
four_months = 4*8*30
two_months = 2*8*30
months = 8*30
two_weeks = 8*14
weeks = 8*7
# two_days = 8*2
# days = 8
# hours6 = 2
# hours3 = 1

# Time interval selected
time_int = two_weeks

param_list = ['Hs', 'Fp', 'Tp', 'Dp']

unit_list = ['m', 'Hz', 's', '°']

tl_x_list = [np.arange(0,9,1), 
             np.arange(int(np.min(ndbc_51101_df['Fp'])),int(np.max(ndbc_51101_df['Fp']))+0.1,0.001),
             np.arange(int(np.min(ndbc_51101_df['Tp']))-2,int(np.max(ndbc_51101_df['Tp']))+2,1),
             np.arange(0,360+15,15)]

stats_dict = {}
stats_table_dict = {}

for param in param_list:
    
    idx = param_list.index(param)
    
    x = ndbc_51001_df[param]
    y = waverys_hawaii_df[param]
    
    slope, intercept, rvalue, _, _ = wv_stats.r_coefficient(x, y)
    
    stats = {'BIAS': wv_stats.bias(x, y),
            'NBIAS': wv_stats.norm_bias(x, y),
            'RMSE': wv_stats.rmse(x, y),
            'NRMSE': wv_stats.norm_rmse(x, y),
            'r': rvalue,
            'R2': wv_stats.r2_coefficient(x, y),
            'SI': wv_stats.scatter_index(x, y)}
    
    stats_dict[param] = {'linear_reg': {
                         'slope': slope,
                         'intercept': intercept,
                         'rvalue': rvalue},
                         'stats': stats,
                         'tendency_x': wv_stats.tendency_line(x, y, tl_x_list[idx])[0],
                         'tendency_y': wv_stats.tendency_line(x, y, tl_x_list[idx])[1],}
    
    stats_table_dict[param] = (
        f'Bias = {np.round(stats["BIAS"],3)}{unit_list[idx]}\n'
        f'NBias = {np.round(stats["NBIAS"],3)}\n'
        f'RMSE = {np.round(stats["RMSE"],3)}{unit_list[idx]}\n'
        f'NRMSE = {np.round(stats["NRMSE"],3)}\n'
        f'S.I. = {np.round(stats["SI"],3)}\n'
        f'r = {np.round(stats["r"],3)}\n'
        f'R² = {np.round(stats["R2"],3)}\n'
        f'y = {np.round(slope,3)}x + {np.round(intercept,3)}\n'
        f'N = {len(x)}')


############################ plot settings ####################################

# param_list = ['Hs', 'Fp', 'Tp', 'Dp']

type_list = ['wave_height', 'frequency', 'period', 'direction']

colors_list = ['royalblue', 'purple', 'tomato', 'lime']



for param in param_list:
    
    idx = param_list.index(param)
    
    wv_val.scatter_plot(ndbc_51001_df[param], waverys_hawaii_df[param], 
                        type_list[idx], colors_list[idx], 'o', 30, True, 
                        stats_dict[param]['tendency_x'],  
                        stats_dict[param]['tendency_y'], 
                        f'{param} - NDBC vs WAVERYS', 'NDBC', 'WAVERYS', True, 
                        stats_table_dict[param], 
                        f'{param}_NDBC_WAVERYS_scatter_plot')    
    
    
