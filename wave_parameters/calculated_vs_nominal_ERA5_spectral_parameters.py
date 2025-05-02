#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Script: calculated_vs_nominal_ERA5_spectral_parameters.py

# Author: Yuri Brasil - yuri.brasil@oceanica.ufrj.br

# Created on Sun Nov 26 17:17:10 2023

# Modification: May 2, 2025

# Objective: Reading of spectra and calculation of wave spectral parameters
             using the spec_parameters package.

"""

# Observations:
    
# The native grid is the reduced latitude/longitude grid of 0.36 degrees 
# (1.0 degree for the EDA)

# For ERA, because there are a total of 24 directions, the direction increment 
# is 15 degrees with the first direction given by half the increment, 
# namely 7.5 degree, where direction 0. means going towards the north and 90 
# towards the east (Oceanographic convention), or more precisely, this 
# should be expressed in gradient since the spectra are in m^2 /(Hz radian).
# The first frequency is 0.03453 Hz and the following ones are: 
# f(n) = f(n-1)*1.1, n=2,30

# The units are degrees true, which means the direction relative to the
# geographic location of the north pole. It is the direction that waves are
# coming from, so 0 degrees means "coming from the north" and 90 degrees means 
# "coming from the east".

import locale
import numpy as np
import xarray as xr
import matplotlib as mpl
from matplotlib import style
from datetime import datetime
import wave_spectral_parameters as par
import matplotlib.pyplot as plt
import wave_statistical_parameters as wave_stats

# Getting the start time
start_time = datetime.now()

########################## Setting Figure parameters #########################

#mpl.rcParams['figure.figsize'] = (10, 7)
mpl.rcParams['figure.figsize'] = (10, 4.8)
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.dpi'] = 300

# Don't cut nothing in the figure saved
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'

mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.edgecolor'] = 'black' 

# Style
style.use('ggplot')

######################## Reading the ERA5 files ###############################

# Reading the netCDF file
nc_name_spec = 'wave_spectra_march_2022_dowload.nc'
nc_name_param = 'wave_parameters_march_2022_download.nc'

my_spec_file = xr.open_dataset(nc_name_spec)  
my_param_file = xr.open_dataset(nc_name_param)  

##################### Getting the variables ###################################

# Reading the longitude and latitude vectors
lon_spec_vec = my_spec_file['longitude'].values
lat_spec_vec = my_spec_file['latitude'].values
lon_param_vec = my_param_file['longitude'].values
lat_param_vec = my_param_file['latitude'].values

# Get the indexes of longitude and latitude
[idx_lon1] = np.where(lon_spec_vec == -44.0)
[idx_lon2] = np.where(lon_param_vec == -44.0)
[idx_lat1] = np.where(lat_spec_vec == -24.0)
[idx_lat2] = np.where(lat_param_vec == -24.0)

# Get the coordinates
lon_spec = idx_lon1[0]
lon_param = idx_lon2[0]
lat_spec = idx_lat1[0]
lat_param = idx_lat2[0]

# Treating the direction converting from "where they come from" to 
# "where they go to"
mean_dir = my_param_file['mwd'].values[:,lat_param,lon_param]
mean_dir = mean_dir-180
mean_dir = np.where(mean_dir<0,mean_dir+360, mean_dir)

# Nominal dictionary
nominal_parameters_dict = {'Hs':my_param_file['swh'].values[:,lat_param,lon_param],
                           'Dm':mean_dir,
                           'Tm':my_param_file['mwp'].values[:,lat_param,lon_param],
                           'Tp':my_param_file['pp1d'].values[:,lat_param,lon_param]}


############################# Time variables ##################################

# Set the locale to English
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')

# datetime strings
date_spec_strings = my_spec_file['valid_time'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist()
date_param_strings = my_param_file['valid_time'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist()

day_strings = my_param_file['valid_time'].dt.strftime('%d').values.tolist()

# Time interval selected
time_int = 24

############## Creating the frequency and direction vectors ###################
    
# Frequency vector size
freq_size = len(my_spec_file['frequencyNumber'])

# Direction vector size
dir_size = len(my_spec_file['directionNumber'])

# Frequency increment
freq_inc = 0.1

# Creating an empty array with the same length of frequency resolution
freq_vec = np.zeros(freq_size)

# First frequency bin
freq_vec[0] = 0.03453

# Generating the frequency vector
for x in np.arange(1,freq_size): freq_vec[x]=freq_vec[x-1]*(1.0+freq_inc)

# Generating the direction vector
dir_step = 15
first_dir = 7.5
dir_vec = np.arange(first_dir,dir_size*dir_step,dir_step)

# Frequency increment and df
increment = (freq_vec[2]/freq_vec[1]) - 1
df = par.df_logarithmic(freq_vec, increment)

# Spectra
#(time, frequency, direction, latitude, longitude) > 24°S and 44°W
spectra = my_spec_file['d2fd'].values[:,:,:,lat_spec,lon_spec] 

# Spectra matrices
spec = 10**spectra[:,:,:]
spec2d = np.nan_to_num(spec, nan=0.0)

############## Calculating wave parameters from spectra #######################

param_keys = ['Hs','Tp','Tm','Dm']

spec_hs = []
spec_dm = []
spec_tm = []
spec_tp = []

for i in range(len(date_spec_strings)):
    
    my_hs = par.hs_spec(freq_vec, df, dir_vec, spec2d[i,:,:], True)
    my_dm = par.mean_direction(freq_vec, df, dir_vec, spec2d[i,:,:], True)
    my_tm, my_fm, my_lm = par.mean_frequency(freq_vec, df, dir_vec, spec2d[i,:,:], True, -1)
    my_fp, my_tp, my_lp = par.peak_frequency(freq_vec, dir_vec, spec2d[i,:,:], True)    
    
    spec_hs.append(my_hs)
    spec_dm.append(my_dm)
    spec_tm.append(my_tm)
    spec_tp.append(my_tp)
        
# Param dictionaries
calculated_parameters_dict = {'Hs':spec_hs,
                              'Dm':spec_dm,
                              'Tm':spec_tm,
                              'Tp':spec_tp}

########################### Preparing the Plot ################################

# Hs ticks
hs_max = 3.5
hs_step = 0.5
hs_ticks = np.arange(0.0, hs_max + hs_step ,hs_step)

# Hs tick labels
hs_tick_labels = []           
for h in range(len(hs_ticks)):
    if (h % 2) == 0:
        hs_tick_labels.append(str(hs_ticks[h])+'m')
    else:
        hs_tick_labels.append(' ')

hs_tick_labels[0] = ' '

# Tp and Tm ticks
t_max = 16
t_step = 1
t_ticks = np.arange(4, t_max + t_step ,t_step)

# Tp and Tm tick labels
t_tick_labels = []           
for t in range(len(t_ticks)):
    if (t % 2) == 0:
        t_tick_labels.append(str(t_ticks[t])+'s')
    else:
        t_tick_labels.append(' ')

t_tick_labels[0] = ' '

# Dp and Dm ticks
t_ticks = np.arange(4, t_max + t_step ,t_step)

# Creating the Direction ticks
d_ticks = np.arange(0,375,15)

d_tick_labels = []
for d in range(len(d_ticks)):
    if (d % 2) == 0:
        d_tick_labels.append(str(d_ticks[d])+'°')
    else:
        d_tick_labels.append(' ')

d_tick_labels[0] = ' '

########################### Parameters list ###################################

# LIst of filenames
filename_string_list = ['hs_comparison_ERA5', 'tp_comparison_ERA5', 
                        'tm_comparison_ERA5', 'dm_comparison_ERA5']

# List of tick vectors
parameter_ticks_list = [hs_ticks, t_ticks, t_ticks, d_ticks]

# List of tick labels vectors
parameter_tick_labels_list = [hs_tick_labels, t_tick_labels, t_tick_labels, d_tick_labels]

# List of colors
colors_list = ['royalblue', 'tomato', 'magenta', 'lime']

############################ Statistics loop ##################################

# List of units
unit_list = ['m','s','s','°']

# Initialize a dictionary to store statistical results
stats_dict = {}

# Compute metrics for each parameter
for k in param_keys:
    
    # Index of k
    idx = param_keys.index(k)
    
    # Nominal and calculated parameters
    nominal = nominal_parameters_dict[k]
    calculated = calculated_parameters_dict[k]
    
    # Wave statistical parameters
    slope, intercept, rvalue, pvalue, _ = wave_stats.r_coefficient(nominal, calculated)
    
    # Tendency lines
    tl_x, tl_y = wave_stats.tendency_line(nominal, calculated, parameter_ticks_list[idx])
    
    # Statistics
    stats_dict[k] = {'bias': wave_stats.bias(nominal, calculated),
                     'rmse': wave_stats.rmse(nominal, calculated),
                     'scatter_index': wave_stats.scatter_index(nominal, calculated),
                     'rvalue':rvalue,
                     'pvalue':pvalue,
                     'slope':slope,
                     'intercept':intercept,
                     'tendency_line_x':tl_x,
                     'tendency_line_y':tl_y,
                     'r2':wave_stats.r2_coefficient(nominal, calculated)}
    
############################### Plot loop #####################################

# for t in range(len(date_param_strings)):
for p in range(len(param_keys)):
    
    # Parameter ticks
    parameter_ticks = parameter_ticks_list[p]
    parameter_tick_labels = parameter_tick_labels_list[p]
    parameter_tick_step = parameter_ticks[1]-parameter_ticks[0]

    ########################### Time series plot ##############################

    # figure 1
    fig, ax = plt.subplots()           
    
    # Plotting the nominal wave parameters
    plt.plot(date_param_strings, nominal_parameters_dict[param_keys[p]], 
             color='black', label=f'Nominal {param_keys[p]}',
                linewidth=0.8, zorder=20)
    
    # Plotting the calculated wave parameters
    plt.plot(date_param_strings, calculated_parameters_dict[param_keys[p]], 
             color=colors_list[p], label=f'Calculated {param_keys[p]}',
                linewidth=0.8, zorder=20)

    # Legend
    leg = plt.legend(loc='upper right', fontsize=10, facecolor='white', 
                     edgecolor='black')
    for text in leg.get_texts():
        text.set_color('black')
    
    # Customizing the ticks
    plt.xticks(date_param_strings[::time_int], labels=day_strings[::time_int], 
               fontsize=6, fontweight='bold')
    plt.yticks(parameter_ticks, labels=parameter_tick_labels, 
               fontweight='bold', fontsize=8)
    ax.tick_params(axis='both',length=0)

    # Setting the parameters of labels
    plt.xlim([date_param_strings[0], date_param_strings[-1]]) 
    plt.ylim([parameter_ticks[0], parameter_ticks[-1] + parameter_tick_step])
    plt.tick_params(length=0)

    # # Adding grid
    # ax.grid(which='major', color='black', alpha=0.2, linestyle='dotted')

    # Fig labels        
    plt.xlabel('Day of March', fontsize=10, fontweight='bold')
    plt.title('Comparison between ERA5 nominal and \n calculated spectral parameters', 
              fontsize=12, fontweight='bold', y=1.0)

    # Showing and saving the plot
    plt.show()
    plt.savefig(f'{filename_string_list[p]}_timeseries_plot', pad_inches=0.1)
    plt.close()

    ############################# Scatter plot ################################
    
    mpl.rcParams['figure.figsize'] = (7, 7)

    # figure 2
    fig, ax = plt.subplots()      
    
    # Scatter plot
    plt.scatter(nominal_parameters_dict[param_keys[p]], 
                calculated_parameters_dict[param_keys[p]], 
                marker='o', s=30, linewidth=0.7, color=colors_list[p], 
                edgecolor='black', alpha=0.6, zorder=3)
    
    # Plotting the identity line
    plt.plot([parameter_ticks[0], parameter_ticks[-1]], 
             [parameter_ticks[0], parameter_ticks[-1]], 
             linestyle='--', linewidth=1.5, 
             color='black', zorder=2)
    
    # Plotting the tendency line
    plt.plot(stats_dict[param_keys[p]]['tendency_line_x'], 
             stats_dict[param_keys[p]]['tendency_line_y'],
             linestyle='--', linewidth=1.5, color=colors_list[p], zorder=2)
        
    # Customizing the ticks
    plt.xticks(parameter_ticks, labels=parameter_tick_labels, 
               fontweight='bold', fontsize=9)
    plt.yticks(parameter_ticks, labels=parameter_tick_labels, 
               fontweight='bold', fontsize=9)
    ax.tick_params(axis='both',length=0)
    
    # Setting x and y limits
    plt.xlim(xmin=parameter_ticks[0], xmax=parameter_ticks[-1])
    plt.ylim(ymin=parameter_ticks[0], ymax=parameter_ticks[-1])

    # Setting the X label
    plt.xlabel(f'{param_keys[p]} - Nominal parameter', fontsize=12, fontweight='bold', 
               zorder=2, color='black')
    plt.ylabel(f'{param_keys[p]} - Calculated parameter', fontsize=12, fontweight='bold', 
               zorder=2, color='black')    
    
    # Setting the Title
    plt.title('Comparison between ERA5 nominal and \n calculated parameter', 
              fontsize=13, fontweight='bold', color='black', y=1.0)    

    x_pos = (len(parameter_ticks)/8)/len(parameter_ticks)
    y_pos = (6*len(parameter_ticks)/8)/len(parameter_ticks)

    # Creating a box to display the statistical parameters
    plt.figtext(x_pos, y_pos,
                'Bias = ' + str(np.round(stats_dict[param_keys[p]]['bias'],2)) + 
                            unit_list[p] + '\n'
                'RMSE = ' + str(np.round(stats_dict[param_keys[p]]['rmse'],2)) + 
                            unit_list[p] + '\n'          
                'S.I. = ' + str(np.round(stats_dict[param_keys[p]]['scatter_index'],2)) + '\n'               
                'p = ' + str(round(stats_dict[param_keys[p]]['pvalue'],4)) + '\n'
                'r = ' + str(round(stats_dict[param_keys[p]]['rvalue'],3)) + '\n'
                'R² = ' + str(round(stats_dict[param_keys[p]]['r2'],3)) + '\n'
                'y = ' + str(round(stats_dict[param_keys[p]]['slope'],2)) +  'x' + \
                ' + ' + str(round(stats_dict[param_keys[p]]['intercept'],2)) + '\n'                
                'N = ' + str(len(nominal_parameters_dict[param_keys[p]])),
                fontsize=8, color='black', 
                bbox=dict(facecolor='white', alpha=1, edgecolor='black'))
        
    # Showing and saving the plot
    plt.show()
    plt.savefig(f'{filename_string_list[p]}_scatter_plot', pad_inches=0.1)
    plt.close()    
        
# Getting the end time and printing running time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
       
        
       
    
        
       
        
       
        
       
        
       