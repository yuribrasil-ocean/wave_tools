#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:08:12 2024

# Script: repsol_pnboia_data_plot.py

# Author: Yuri Brasil

# e-mail: yuri.brasil@oceanica.ufrj.br

# Modification: May 8th 2024

# Objective: Plot a timeseries of Hs, Tp and Dp from PNBOIA buoys to be used 
             for validation

"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style

from datetime import datetime
from datetime import timedelta
import matplotlib.ticker as ticker

########################## Setting Figure parameters #########################

#mpl.rcParams['figure.figsize'] = (10, 7)
mpl.rcParams['figure.figsize'] = (10, 4.8)
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.dpi'] = 300

# Don't cut nothing in the figure saved
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'

style.use('ggplot')

mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.edgecolor'] = 'black' 

##################### Reading csv files as DataFrames #########################

# Main path
# main_path = '/run/media/numa20'
main_path = '/media/yuri'

# NDBC 51001
ndbc_51001_df = pd.read_csv('ndbc_51001_df_final.csv', index_col=[0]) 

# NDBC 51001
ndbc_51101_df = pd.read_csv('ndbc_51101_df_final.csv', index_col=[0]) 

# WAVERYS
waverys_hawaii_df = pd.read_csv('waverys_hawaii_point_df.csv', index_col=[0]) 

# Replacing -9999 by np.NaN

my_dfs = []

for df in [ndbc_51001_df, ndbc_51101_df, waverys_hawaii_df]:
    df['Hs'] = np.where(df['Hs'] == -9999, np.NaN, df['Hs'])
    df['Tp'] = np.where(df['Tp'] == -9999, np.NaN, df['Tp'])
    df['Dp'] = np.where(df['Dp'] == -9999, np.NaN, df['Dp'])

    # # Quality control
    # hs_mean = np.mean(df['Hs'])
    # hs_std = np.std(df['Hs'])
    # tp_mean = np.mean(df['Tp'])
    # tp_std = np.std(df['Tp'])
    # dp_mean = np.mean(df['Dp'])
    # dp_std = np.std(df['Dp'])


    my_dfs.append(df)

title_list = ['NDBC 51001', 'NDBC 51101', 'WAVERYS']

buoys_str = ['ndbc_51001','ndbc_51101','waverys_hawaii']

dict_dfs = {buoys_str[0]:my_dfs[0],
            buoys_str[1]:my_dfs[1],
            buoys_str[2]:my_dfs[2]}

######################### Preparing the Plot ##################################

# Creating strings for the yticks
Hs_ticks = ['', '1m', '2m', '3m', '4m', '5m','6m','7m','8m']
Tp_ticks = ['', '2s', '4s', '6s', '8s', '10s', '12s',
            '14s', '16s', '18s', '20s', '22s']

# Creating the Direction ticks
Dp_ticks = []
for i in range(-30,375,15):
    Dp_ticks.append(str(i)+'°')

Dp_ticks[0] = ''

# Removing the ones which ends with 5º
for i in range(-1,24,2):
    Dp_ticks[i+2] = ''

x_pos = 0.055
y_pos = 0.83

#################### Preparing the datetime strings ###########################

# date_strings = ndbc_51001_df.index.dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist()

my_dates = pd.to_datetime(ndbc_51001_df.index)
datetime_list = my_dates.to_list()


day_month_list = []

for d in datetime_list:
    
    # my_datetime = datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
    
    day_month_list.append(d.strftime('%d/%m' ))


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

param_list = ['Hs', 'Tp', 'Dp']

# Initialize the dictionaries
mean_dict = {}
std_dict = {}
max_dict = {}

# Loop through each parameter in param_list
for param in param_list:
    
    # Initialize the second-level dictionary for each parameter
    mean_dict[param] = {}
    std_dict[param] = {}
    max_dict[param] = {}
    
    # Loop through each buoy in buoys_str
    for buoy in buoys_str:
        
        # Calculate the mean for the current parameter and buoy
        mean_value = np.mean(dict_dfs[buoy][param])
        
        # Assign the calculated mean value to the dictionary
        mean_dict[param][buoy] = mean_value
        
        # Calculate the standard deviation for the current parameter and buoy
        std_value = np.std(dict_dfs[buoy][param])
        
        # Assign the calculated standard deviation value to the dictionary
        std_dict[param][buoy] = std_value
        
        # Calculate the maximum value for the current parameter and buoy
        max_value = np.max(dict_dfs[buoy][param])
        
        # Assign the calculated maximum value to the dictionary
        max_dict[param][buoy] = max_value
        

ndbc_51001_df = pd.read_csv('ndbc_51001_df_final.csv', index_col=[0]) 

# NDBC 51001
ndbc_51101_df = pd.read_csv('ndbc_51101_df_final.csv', index_col=[0]) 
        
buoys_str = ['ndbc_51001','ndbc_51101','waverys_hawaii']
title_list = ['NDBC 51001', 'NDBC 51101', 'WAVERYS']

############################## The Plot - Hs ##################################

# Setting the color backgroung
fig, ax1 = plt.subplots()

# NDBC 51001 Line plot
plt.plot(dict_dfs['ndbc_51001'].index, dict_dfs['ndbc_51001']['Hs'], '-', linewidth=1.0, 
          color='navy', zorder=3, label='NDBC 51001')

# NDBC 51101 Line plot
plt.plot(dict_dfs['ndbc_51101'].index, dict_dfs['ndbc_51101']['Hs'], '-', linewidth=1.0, 
          color='cyan', zorder=3, label='NDBC 51101')

# WAVERYS Line plot
plt.plot(dict_dfs['waverys_hawaii'].index, dict_dfs['waverys_hawaii']['Hs'], '-', linewidth=1.0, 
          color='black', zorder=3, label='WAVERYS')

# Plotting the NDBC 51001 mean line
plt.hlines(mean_dict['Hs']['ndbc_51001'], dict_dfs['ndbc_51001'].index[0], 
           dict_dfs['ndbc_51001'].index[-1], linestyle='--',
           linewidth=2.0, color='navy')

# Plotting the NDBC 51101 mean line
plt.hlines(mean_dict['Hs']['ndbc_51101'], dict_dfs['ndbc_51101'].index[0], 
           dict_dfs['ndbc_51101'].index[-1], linestyle='--',
           linewidth=2.0, color='cyan')

# Plotting the WAVERYS mean line
plt.hlines(mean_dict['Hs']['waverys_hawaii'], dict_dfs['waverys_hawaii'].index[0], 
           dict_dfs['waverys_hawaii'].index[-1], linestyle='--',
           linewidth=2.0, color='black')

# Legend
leg = plt.legend(loc='upper center', fontsize=8, facecolor='white', edgecolor='black')
for text in leg.get_texts():
    text.set_color('black')

# Customizing the ticks
plt.xticks(dict_dfs['ndbc_51001'].index[::time_int], labels=day_month_list[::time_int],fontsize=7, fontweight='bold')
plt.yticks(np.arange(0,9,1), labels=Hs_ticks, fontweight='bold', fontsize=7)
ax1.tick_params(axis='both',length=0)

# Setting x and y limits
plt.xlim(xmin=dict_dfs['ndbc_51001'].index[0], xmax=dict_dfs['ndbc_51001'].index[-1])
plt.ylim(ymin=0, ymax=8)

# Setting the X label
plt.xlabel('Data (mês/ano)', fontsize=11, fontweight='bold', zorder=2, color='black')

#Setting the Title  
beg = str(dict_dfs['ndbc_51001'].index[0])
end = str(dict_dfs['ndbc_51001'].index[len(dict_dfs['ndbc_51001'])-1])
my_title = f'Altura Significativa (Hs) - De {beg} até {end}'

plt.title(my_title, fontsize=10, fontweight='bold', color='black')

# Creating a box to display the statistical parameters
plt.figtext(x_pos, y_pos,
            'Max = ' + str(np.round(max_dict['Hs']['waverys_hawaii'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Hs']['waverys_hawaii'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Hs']['waverys_hawaii'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['waverys_hawaii']['Hs'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='black', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

plt.figtext(x_pos+0.7, y_pos,
            'Max = ' + str(np.round(max_dict['Hs']['ndbc_51001'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Hs']['ndbc_51001'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Hs']['ndbc_51001'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['ndbc_51001']['Hs'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='navy', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

plt.figtext(x_pos+0.805, y_pos,
            'Max = ' + str(np.round(max_dict['Hs']['ndbc_51101'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Hs']['ndbc_51101'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Hs']['ndbc_51101'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['ndbc_51101']['Hs'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='cyan', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))


#plt.show()

fig.savefig('Hs_buoys_waverys_timeseries_ggplot')

############################## The Plot - Tp ##################################

# Setting the color backgroung
fig, ax2 = plt.subplots()

# NDBC 51001 Line plot
plt.plot(dict_dfs['ndbc_51001'].index, dict_dfs['ndbc_51001']['Tp'], '-', linewidth=1.0, 
          color='tomato', zorder=3, label='NDBC 51001')

# NDBC 51101 Line plot
plt.plot(dict_dfs['ndbc_51101'].index, dict_dfs['ndbc_51101']['Tp'], '-', linewidth=1.0, 
          color='orange', zorder=3, label='NDBC 51101')

# WAVERYS Line plot
plt.plot(dict_dfs['waverys_hawaii'].index, dict_dfs['waverys_hawaii']['Tp'], '-', linewidth=1.0, 
          color='black', zorder=3, label='WAVERYS')

# Plotting the NDBC 51001 mean line
plt.hlines(mean_dict['Tp']['ndbc_51001'], dict_dfs['ndbc_51001'].index[0], 
           dict_dfs['ndbc_51001'].index[-1], linestyle='--',
           linewidth=2.0, color='tomato')

# Plotting the NDBC 51101 mean line
plt.hlines(mean_dict['Tp']['ndbc_51101'], dict_dfs['ndbc_51101'].index[0], 
           dict_dfs['ndbc_51101'].index[-1], linestyle='--',
           linewidth=2.0, color='orange')

# Plotting the WAVERYS mean line
plt.hlines(mean_dict['Tp']['waverys_hawaii'], dict_dfs['waverys_hawaii'].index[0], 
           dict_dfs['waverys_hawaii'].index[-1], linestyle='--',
           linewidth=2.0, color='black')

# Legend
leg = plt.legend(loc='upper center', fontsize=8, facecolor='white', edgecolor='black')
for text in leg.get_texts():
    text.set_color('black')

# Customizing the ticks
plt.xticks(dict_dfs['ndbc_51001'].index[::time_int], labels=day_month_list[::time_int],fontsize=7, fontweight='bold')
plt.yticks(np.arange(0,24,2), labels=Tp_ticks, fontweight='bold', fontsize=7)
ax2.tick_params(axis='both',length=0)    

# Setting x and y limits
plt.xlim(xmin=dict_dfs['ndbc_51001'].index[0], xmax=dict_dfs['ndbc_51001'].index[-1])
plt.ylim(ymin=0, ymax=24)

# Setting the X label
plt.xlabel('Data (mês/ano)', fontsize=11, fontweight='bold', zorder=2, color='black')

#Setting the Title  
beg = str(dict_dfs['ndbc_51001'].index[0])
end = str(dict_dfs['ndbc_51001'].index[len(dict_dfs['ndbc_51001'])-1])
my_title = f'Periodo de Pico (Tp) - De {beg} até {end}'

plt.title(my_title, fontsize=10, fontweight='bold', color='black')

# Creating a box to display the statistical parameters
plt.figtext(x_pos, y_pos,
            'Max = ' + str(np.round(max_dict['Tp']['waverys_hawaii'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Tp']['waverys_hawaii'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Tp']['waverys_hawaii'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['waverys_hawaii']['Tp'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='black', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

plt.figtext(x_pos+0.7, y_pos,
            'Max = ' + str(np.round(max_dict['Tp']['ndbc_51001'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Tp']['ndbc_51001'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Tp']['ndbc_51001'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['ndbc_51001']['Tp'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='tomato', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

plt.figtext(x_pos+0.805, y_pos,
            'Max = ' + str(np.round(max_dict['Tp']['ndbc_51101'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Tp']['ndbc_51101'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Tp']['ndbc_51101'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['ndbc_51101']['Tp'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='orange', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))



fig.savefig('Tp_buoys_waverys_timeseries_ggplot')

########################## The Plot - Mean Direction ##########################

# Setting the color backgroung
fig, ax3 = plt.subplots()

# NDBC 51001 Line plot
plt.plot(dict_dfs['ndbc_51001'].index, dict_dfs['ndbc_51001']['Dp'], '-', linewidth=1.0, 
          color='olive', zorder=3, label='NDBC 51001')

# NDBC 51101 Line plot
plt.plot(dict_dfs['ndbc_51101'].index, dict_dfs['ndbc_51101']['Dp'], '-', linewidth=1.0, 
          color='lawngreen', zorder=3, label='NDBC 51101')

# WAVERYS Line plot
plt.plot(dict_dfs['waverys_hawaii'].index, dict_dfs['waverys_hawaii']['Dp'], '-', linewidth=1.0, 
          color='black', zorder=3, label='WAVERYS')

# Plotting the NDBC 51001 mean line
plt.hlines(mean_dict['Dp']['ndbc_51001'], dict_dfs['ndbc_51001'].index[0], 
           dict_dfs['ndbc_51001'].index[-1], linestyle='--',
           linewidth=2.0, color='olive')

# Plotting the NDBC 51101 mean line
plt.hlines(mean_dict['Dp']['ndbc_51101'], dict_dfs['ndbc_51101'].index[0], 
           dict_dfs['ndbc_51101'].index[-1], linestyle='--',
           linewidth=2.0, color='lawngreen')

# Plotting the WAVERYS mean line
plt.hlines(mean_dict['Dp']['waverys_hawaii'], dict_dfs['waverys_hawaii'].index[0], 
           dict_dfs['waverys_hawaii'].index[-1], linestyle='--',
           linewidth=2.0, color='black')

# Legend
leg = plt.legend(loc='upper center', fontsize=8, facecolor='white', edgecolor='black')
for text in leg.get_texts():
    text.set_color('black')

# Customizing the ticks
plt.xticks(dict_dfs['ndbc_51001'].index[::time_int], labels=day_month_list[::time_int],fontsize=7, fontweight='bold')
plt.yticks(np.arange(-30,375,15), labels=Dp_ticks, fontweight='bold', fontsize=7)
ax3.tick_params(axis='both',length=0)   
ax3.invert_yaxis()

# Setting x and y limits
plt.xlim(xmin=dict_dfs['ndbc_51001'].index[0], xmax=dict_dfs['ndbc_51001'].index[-1])
plt.ylim(ymin=375, ymax=-30)

# Setting the X label
plt.xlabel('Data (mês/ano)', fontsize=11, fontweight='bold', zorder=2, color='black')

#Setting the Title  
beg = str(dict_dfs['ndbc_51001'].index[0])
end = str(dict_dfs['ndbc_51001'].index[len(dict_dfs['ndbc_51001'])-1])
my_title = f'Direção de Pico (Dp) - De {beg} até {end}'

plt.title(my_title, fontsize=10, fontweight='bold', color='black')

# Creating a box to display the statistical parameters
plt.figtext(x_pos, y_pos,
            'Max = ' + str(np.round(max_dict['Dp']['waverys_hawaii'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Dp']['waverys_hawaii'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Dp']['waverys_hawaii'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['waverys_hawaii']['Dp'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='black', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

plt.figtext(x_pos+0.7, y_pos,
            'Max = ' + str(np.round(max_dict['Dp']['ndbc_51001'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Dp']['ndbc_51001'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Dp']['ndbc_51001'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['ndbc_51001']['Dp'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='olive', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

plt.figtext(x_pos+0.805, y_pos,
            'Max = ' + str(np.round(max_dict['Dp']['ndbc_51101'],2)) + 'm' + '\n'
            'Média = ' + str(np.round(mean_dict['Dp']['ndbc_51101'],2)) + 'm' + '\n'
            'Desvio = '+ str(np.round(std_dict['Dp']['ndbc_51101'],2)) + 'm' + '\n'
            'N = ' + str(dict_dfs['ndbc_51101']['Dp'].count()),
            # 'N = ' + str(len(dict_dfs[buoy_str])),
            fontsize=6, color='lawngreen', 
            bbox=dict(facecolor='white', alpha=1, edgecolor='black'))



fig.savefig('Dp_buoys_waverys_timeseries_ggplot')











# ############################## The Plot - Tp ##################################

# # Setting the color backgroung
# fig, ax2 = plt.subplots()

# # Buoy Line plot
# plt.plot(dict_dfs[buoy_str].index, dict_dfs[buoy_str]['Tp'], '-', linewidth=1.0, 
#           color='tomato', zorder=3, label=title_list[my_index])

# # Plotting the buoy mean line
# plt.hlines(Tp_data_mean, dict_dfs[buoy_str].index[0], dict_dfs[buoy_str].index[-1], linestyle='--',
#            linewidth=2.0, color='black')


# # Legend
# leg = plt.legend(loc='upper center', fontsize=10, facecolor='white', edgecolor='black')
# for text in leg.get_texts():
#     text.set_color('black')
    
# # Customizing the ticks
# plt.xticks(dict_dfs[buoy_str].index[::time_int], labels=day_month_list[::time_int],fontsize=7, fontweight='bold')
# plt.yticks(np.arange(0,24,2), labels=Tp_ticks, fontweight='bold', fontsize=7)
# ax2.tick_params(axis='both',length=0)    

# # Setting x and y limits
# plt.xlim(xmin=dict_dfs[buoy_str].index[0], xmax=dict_dfs[buoy_str].index[-1])
# plt.ylim(ymin=0, ymax=24)

# # Setting the X label
# plt.xlabel('Data', fontsize=11, fontweight='bold', zorder=2, color='black')

# #Setting the Title  
# beg = str(dict_dfs[buoy_str].index[0])
# end = str(dict_dfs[buoy_str].index[len(dict_dfs[buoy_str])-1])
# my_title = f'Boia {title_list[my_index]} - Período de Pico - De {beg} até {end}'

# plt.title(my_title, fontsize=10, fontweight='bold', color='black')

# # Creating a box to display the statistical parameters
# plt.figtext(x_pos, y_pos,
#             'Max = ' + str(np.round(Tp_data_max,2)) + 's' + '\n'
#             'Média = ' + str(np.round(Tp_data_mean,2)) + 's' + '\n'
#             'Desvio = '+ str(np.round(Tp_data_std,2)) + 's' + '\n'
#             'N = ' + str(dict_dfs[buoy_str]['Tp'].count()),
#             # 'N = ' + str(len(dict_dfs[buoy_str])),
#             fontsize=8, color='black', 
#             bbox=dict(facecolor='white', alpha=1, edgecolor='black'))


# #plt.show()

# fig.savefig(f'Tp_{buoys_str[my_index]}_timeseries_ggplot')

# ########################## The Plot - Mean Direction ##########################

# # Setting the color backgroung
# fig, ax3 = plt.subplots()

# # Buoy Line plot
# plt.plot(dict_dfs[buoy_str].index, dict_dfs[buoy_str]['Dp'], '-', linewidth=1.0, 
#           color='seagreen', zorder=3, label=title_list[my_index])

# # Plotting the buoy mean line
# plt.hlines(Dp_data_mean, dict_dfs[buoy_str].index[0], dict_dfs[buoy_str].index[-1], linestyle='--',
#            linewidth=2.0, color='black')    

# # Legend
# leg = plt.legend(loc='upper center', fontsize=10, facecolor='white', edgecolor='black')
# # Make the font colors equal to the lines
# for line, text in zip(leg.get_lines(), leg.get_texts()):
#     text.set_color(line.get_color())

# # Customizing the ticks
# plt.xticks(dict_dfs[buoy_str].index[::time_int], labels=day_month_list[::time_int],fontsize=7, fontweight='bold')
# plt.yticks(np.arange(-30,375,15), labels=Dp_ticks, fontweight='bold', fontsize=7)
# ax3.tick_params(axis='both',length=0)   
# ax3.invert_yaxis()

# # Setting x and y limits
# plt.xlim(xmin=dict_dfs[buoy_str].index[0], xmax=dict_dfs[buoy_str].index[-1])
# plt.ylim(ymin=375, ymax=-30)

# # Setting the X label
# plt.xlabel('Data', fontsize=11, fontweight='bold', zorder=2, color='black')

# #Setting the Title  
# beg = str(dict_dfs[buoy_str].index[0])
# end = str(dict_dfs[buoy_str].index[len(dict_dfs[buoy_str])-1])
# my_title = f'Boia {title_list[my_index]} - Direção Média - De {beg} até {end}'

# plt.title(my_title, fontsize=10, fontweight='bold', color='black')

# # Creating a box to display the statistical parameters
# plt.figtext(x_pos + 0.01, y_pos,
#             'Max = ' + str(np.round(Dp_data_max,2)) + '°' + '\n'
#             'Média = ' + str(np.round(Dp_data_mean,2)) + '°' + '\n'
#             'Desvio = '+ str(np.round(Dp_data_std,2)) + '°' + '\n'
#             'N = ' + str(dict_dfs[buoy_str]['Dp'].count()),
#             # 'N = ' + str(len(dict_dfs[buoy_str])),
#             fontsize=8, color='black', 
#             bbox=dict(facecolor='white', alpha=1, edgecolor='black'))


# #plt.show()

# fig.savefig(f'Dp_{buoys_str[my_index]}_timeseries_ggplot')






plt.close('all')








