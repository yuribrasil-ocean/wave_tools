#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Script: ERA5_spectral_parameters_from_converted_spectra.py

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
import wave_domain_converters as conv
import matplotlib.pyplot as plt
import wave_statistical_parameters as wave_stats

# Getting the start time
start_time = datetime.now()

########################## Setting Figure parameters #########################

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

# List keys
# List of variable names (data variables)
variable_keys = list(my_spec_file.data_vars.keys())

# all keys (data variables + coordinates)
all_keys = list(my_spec_file.variables.keys())

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
                           'Hs_k':my_param_file['swh'].values[:,lat_param,lon_param],
                           'Hs_interp':my_param_file['swh'].values[:,lat_param,lon_param],
                           'Hs_semilog':my_param_file['swh'].values[:,lat_param,lon_param],
                           'Dm':mean_dir, 'Dm_interp':mean_dir, 'Dm_semilog':mean_dir,
                           'Tm':my_param_file['mwp'].values[:,lat_param,lon_param],
                           'Tm_interp':my_param_file['mwp'].values[:,lat_param,lon_param],
                           'Tm_semilog':my_param_file['mwp'].values[:,lat_param,lon_param],
                           'Tp':my_param_file['pp1d'].values[:,lat_param,lon_param],
                           'Tp_interp':my_param_file['pp1d'].values[:,lat_param,lon_param],
                           'Tp_semilog':my_param_file['pp1d'].values[:,lat_param,lon_param]}

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

# Genearating the wavenumber vector
g = 9.80665
L_vec =  g/(2*np.pi*freq_vec**2)
k_vec = (2*np.pi)/L_vec
f_from_k_vec = np.sqrt(g/(L_vec*2*np.pi))

# Generating the direction vector
dir_step = 15
first_dir = 7.5
dir_vec = np.arange(first_dir,dir_size*dir_step,dir_step)

# Frequency and wavenumber increments
f_increment = (freq_vec[2]/freq_vec[1]) - 1
k_increment = (k_vec[2]/k_vec[1]) - 1

# df, dk and dtheta
df = par.d_logarithmic(freq_vec)
dk = par.d_logarithmic(k_vec)
dtheta = np.ones(len(dir_vec))*dir_step

# Spectra
#(time, frequency, direction, latitude, longitude) > 24°S and 44°W
spectra = my_spec_file['d2fd'].values[:,:,:,lat_spec,lon_spec] 
number = my_spec_file['number'].values

# Spectra matrices (transposed)
spec = 10**spectra[:,:,:]
spec2d = np.nan_to_num(spec, nan=0.0)
spec2d = np.moveaxis(spec2d, 0, -1)

############################ New spectrum vectors #############################

# New direction vector and dtheta
dir_vec_new = np.arange(0,360,360/24)
dtheta_new = np.ones(len(dir_vec_new))*360/24

# Creating an empty array with the same length of frequency resolution
freq_vec_new = np.zeros(25)

# First frequency bin
freq_vec_new[0] = 0.044177 #last=0.220300 # DEFAULT

# Generating the new frequency vector
for x in np.arange(1,25): freq_vec_new[x]=freq_vec_new[x-1]*(1.0+0.1)

# Frequency difference vector
df_new = par.d_logarithmic(freq_vec_new)

###############################################################################

# Empty lists to be filled with spectra
spec2d_k = []
spec2d_f_new = []
spec2d_f_new2 = []

# Loop of conversions
for i in range(len(day_strings)):
    
    # Conversion to K domain
    k_spec, k_vec, dir_vec = conv.wave_spec_conversion_f_k(dir_vec, freq_vec,
                                                           spec2d[:,:,i], 
                                                           'to_k')
    
    f_spec = conv.wave_spec_interpolation(dir_vec, freq_vec, spec2d[:,:,i],
                                          dir_vec_new, freq_vec_new, 
                                          log_freq_vec=True, 
                                          interp_method='linear')
    
    f_spec2 = conv.wave_spec_semilog_interpolation(dir_vec, freq_vec, dtheta,
                                                   df, spec2d[:,:,i], 
                                                   dir_vec_new, freq_vec_new, 
                                                   dtheta_new, df_new,
                                                   energy_flag=True)
    
    # Append spectra
    spec2d_k.append(k_spec)
    spec2d_f_new.append(f_spec)
    spec2d_f_new2.append(f_spec2)
    
# Convert list of 2D arrays to a 3D array: [time, f_or_k, θ]
spec2d_k = np.stack(spec2d_k, axis=2)
spec2d_f_new = np.stack(spec2d_f_new, axis=2)
spec2d_f_new2 = np.stack(spec2d_f_new2, axis=2)

############## Calculating wave parameters from spectra #######################

# Parameters keys
param_keys = ['Hs','Hs_k','Hs_interp','Hs_semilog',
              'Tp','Tp_interp', 'Tp_semilog',
              'Tm','Tm_interp', 'Tm_semilog',
              'Dm','Dm_interp', 'Dm_semilog',]

# Empty lists
spec_hs = []
spec_hs_k = []
spec_hs_interp = []
spec_hs_semilog = []

spec_dm = []
spec_dm_interp = []
spec_dm_semilog = []

spec_tm = []
spec_tm_interp = []
spec_tm_semilog = []

spec_tp = []
spec_tp_interp = []
spec_tp_semilog = []

# Loop to save parameters
for i in range(len(date_spec_strings)):
    
    # Hs
    my_hs = par.hs_spec(dir_vec, freq_vec, df, spec2d[:,:,i], True)    
    my_hs_k = par.hs_spec(dir_vec, k_vec, dk, spec2d_k[:,:,i], True)
    my_hs_interp = par.hs_spec(dir_vec_new, freq_vec_new, df_new, spec2d_f_new[:,:,i], True)  
    my_hs_semilog = par.hs_spec(dir_vec_new, freq_vec_new, df_new, spec2d_f_new2[:,:,i], True)  
    
    # Mean direction
    my_dm = par.mean_direction(dir_vec, freq_vec, df, spec2d[:,:,i], True)
    my_dm1 = par.mean_direction(dir_vec_new, freq_vec_new, df_new, spec2d_f_new[:,:,i], True)
    my_dm2 = par.mean_direction(dir_vec_new, freq_vec_new, df_new, spec2d_f_new2[:,:,i], True)
    
    # Mean period
    my_tm, my_fm, my_lm = par.mean_frequency(dir_vec, freq_vec, df, spec2d[:,:,i], True, -1)
    my_tm1, my_fm1, my_lm1 = par.mean_frequency(dir_vec_new, freq_vec_new, df_new, spec2d_f_new[:,:,i], True, -1)
    my_tm2, my_fm2, my_lm2 = par.mean_frequency(dir_vec_new, freq_vec_new, df_new, spec2d_f_new2[:,:,i], True, -1)
    
    # Peak period
    my_fp, my_tp, my_lp = par.peak_frequency(dir_vec, freq_vec, spec2d[:,:,i], True) 
    my_fp1, my_tp1, my_lp1 = par.peak_frequency(dir_vec_new, freq_vec_new, spec2d_f_new[:,:,i], True) 
    my_fp2, my_tp2, my_lp2 = par.peak_frequency(dir_vec_new, freq_vec_new, spec2d_f_new2[:,:,i], True) 
    
    # Append    
    spec_hs.append(my_hs)
    spec_hs_k.append(my_hs_k)
    spec_hs_interp.append(my_hs_interp)
    spec_hs_semilog.append(my_hs_semilog)
    
    spec_dm.append(my_dm)
    spec_dm_interp.append(my_dm1)
    spec_dm_semilog.append(my_dm2)
    
    spec_tm.append(my_tm)
    spec_tm_interp.append(my_tm1)
    spec_tm_semilog.append(my_tm2)
    
    spec_tp.append(my_tp)
    spec_tp_interp.append(my_tp1)
    spec_tp_semilog.append(my_tp2)
        
# Parameters dictionaries
calculated_parameters_dict = {'Hs':spec_hs, 'Hs_k':spec_hs_k, 
                              'Hs_interp':spec_hs_interp, 'Hs_semilog':spec_hs_interp,
                              'Dm':spec_dm, 'Dm_interp':spec_dm_interp, 
                              'Dm_semilog':spec_dm_semilog,
                              'Tm':spec_tm, 'Tm_interp':spec_tm_interp,
                              'Tm_semilog':spec_tm_semilog,
                              'Tp':spec_tp,'Tp_interp':spec_tp_interp,
                              'Tp_semilog':spec_tp_semilog}


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

# List of filenames
filename_string_list = ['hs_comparison_ERA5', 'hs_k_comparison_ERA5',
                        'hs_interp_comparison_ERA5', 'hs_semilog_comparison_ERA5',
                        'tp_comparison_ERA5', 'tp_interp_comparison_ERA5', 
                        'tp_semilog_comparison_ERA5',
                        'tm_comparison_ERA5', 'tm_interp_comparison_ERA5',
                        'tm_semilog_comparison_ERA5', 
                        'dm_comparison_ERA5','dm_interp_comparison_ERA5',
                        'dm_semilog_comparison_ERA5']



# List of tick vectors
parameter_ticks_list = [hs_ticks, hs_ticks, hs_ticks, hs_ticks, 
                        t_ticks, t_ticks, t_ticks,
                        t_ticks, t_ticks, t_ticks,
                        d_ticks, d_ticks, d_ticks]

# List of tick labels vectors
parameter_tick_labels_list = [hs_tick_labels, hs_tick_labels, 
                              hs_tick_labels, hs_tick_labels, 
                              t_tick_labels, t_tick_labels,
                              t_tick_labels,
                              t_tick_labels, t_tick_labels,
                              t_tick_labels,
                              d_tick_labels, d_tick_labels,
                              d_tick_labels]

# List of colors
colors_list = ['cyan', 'tomato', 'magenta', 'lime']

############################ Statistics loop ##################################

# List of units
unit_list = ['m','m','m','m',
             's','s','s',
             's','s','s',
             '°','°','°']

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

    mpl.rcParams['figure.figsize'] = (10, 4.8)

    # figure 1
    fig, ax = plt.subplots()           
    
    # Plotting the nominal wave parameters
    plt.plot(date_param_strings, nominal_parameters_dict[param_keys[p]], 
             color='black', label=f'Nominal {param_keys[p]}',
                linewidth=0.8, zorder=20)
    
    # Plot the calculated wave parameters
    plt.plot(date_param_strings, calculated_parameters_dict[param_keys[p]], 
             color=colors_list[p % len(colors_list)], 
             label=f'Calculated {param_keys[p]}', linewidth=0.8, zorder=20)



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
                marker='o', s=30, linewidth=0.7, 
                color=colors_list[p % len(colors_list)], 
                edgecolor='black', alpha=0.6, zorder=3)
    
    # Plotting the identity line
    plt.plot([parameter_ticks[0], parameter_ticks[-1]], 
             [parameter_ticks[0], parameter_ticks[-1]], 
             linestyle='--', linewidth=1.5, 
             color='black', zorder=2)
    
    # Plot the tendency line
    plt.plot(stats_dict[param_keys[p]]['tendency_line_x'], 
             stats_dict[param_keys[p]]['tendency_line_y'],
             linestyle='--', linewidth=1.5, 
             color=colors_list[p % len(colors_list)], zorder=2)     
       
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
       
        
       