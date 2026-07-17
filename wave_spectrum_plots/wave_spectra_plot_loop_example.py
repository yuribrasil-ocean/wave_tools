#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Script: wave_spectra_plot_loop_example.py

# Author: Yuri Brasil - yuri.brasil@oceanica.ufrj.br

# Created on Sun Nov 26 17:17:10 2023

# Modification: May 5, 2025

# Objective: Read and plot a wave spectrum (f,theta) using several plot 
             functions from wave_spectrum_plots.

# Functions:
            
"""

import os
import numpy as np
import scipy.io as sio
import wave_colormaps as wv_cmap
import wave_spectrum_plots as wv_plt 
import wave_spectral_parameters as wv_par

# Current path
my_path = os.getcwd()

# Reading the file
# my_file = os.path.join(my_path,'data/my_matrices_envisat_25x24_128pts_jonswap_unimodal_nonlinear_3m_45deg_500m.mat')
# my_file = os.path.join(my_path,'data/my_matrices_envisat_161x144_128pts_jonswap_unimodal_nonlinear_3m_45deg_500m.mat')
# my_file = os.path.join(my_path,'data/my_matrices_envisat_1001x720_128pts_jonswap_unimodal_nonlinear_3m_45deg_500m.mat')
my_file = os.path.join(my_path,'data/my_matrices_sentinel_24x25_512pts_jonswap_bimodal_nonlinear_2.5m_45deg_490m_bi_2.0m_135deg_100m.mat')

                       # Read the matfile
output = sio.loadmat(my_file)

# Getting the variables as arrays
freq_vec = output['freq_vec'][0]
dir_vec = output['dir_vec'][0]
spec2d = output['spec_2D']

# Calculate Hs and peak parameters
df = wv_par.d_logarithmic(freq_vec)
hs = wv_par.hs_spec(dir_vec, freq_vec, df, spec2d, True)
fp, tp, lp = wv_par.peak_frequency(dir_vec, freq_vec, spec2d, True)
dp = wv_par.peak_direction(dir_vec, freq_vec, df, spec2d)

# Param dictionary
param_dict = {'Hs':hs,'Fp':fp,'Tp':tp,'Lp':lp,'Dp':dp}

# Geeting the maximum value
maximum_energy = np.max(spec2d)

# set frequency grid values (radius)
ftick = [0.05, 0.10, 0.15, 0.20]

# Flag to plot 0º-180º and 90º-270º Arrows (Polar plot)
arrow_flag = True

# Flag to plot radius labels (Polar plot)
radius_flag = True

# Flag for the normalization 
norm_flag = False

# Flag for mark the spectrum peak
peak_flag = False

# Flag to plot the wave parameters in a text box
param_flag = True

# Colormap for wave spectrum
my_cmap = wv_cmap.wave_spec_cmap

# Size of the ticks
tick_size = 2

for j in ['omni', 'polar', 'cartesian', 'map']:
# for j in ['map']:    
    for k in ['freq', 'per']:
        
        # Flag for the type of plot ('polar', 'cartesian' and 'map')
        plot_flag = j
        
        # Flag for the type of Wave Spectrum ('freq'=(f,Theta) or 'per'=(t,theta))
        vec_flag = k
    
        # Setting figure title
        fig_title = 'Wave Spectrum Density '
        
        # Setting normalization string
        if norm_flag == True:
            norm_str = '_norm'    
        else:    
            norm_str = ''
        
        # Setting the filename    
        fname = f'{plot_flag}_wave_spec_{vec_flag}{norm_str}_int.png'
                
        # The Plots
        
        if plot_flag == 'omni':
            wv_plt.omnidirectional_spec(dir_vec, freq_vec, spec2d, param_dict, 
                                        fname, 'royalblue', norm_flag, peak_flag, 
                                        param_flag, vec_flag, fig_title, tick_size)
        
        if plot_flag == 'polar':
            wv_plt.polar_spec(dir_vec, freq_vec, spec2d, param_dict, fname, 
                              ftick, my_cmap, arrow_flag, radius_flag, 
                              norm_flag, peak_flag, param_flag, vec_flag, 
                              fig_title)
            
        elif plot_flag == 'cartesian':
            wv_plt.cartesian_spec(dir_vec, freq_vec, spec2d, param_dict, fname, 
                                  my_cmap, norm_flag, peak_flag, param_flag, 
                                  vec_flag, fig_title, tick_size)
            
        elif plot_flag == 'map':
            wv_plt.map_spec(dir_vec, freq_vec, spec2d, param_dict, fname, 
                            my_cmap, norm_flag, peak_flag, param_flag, 
                            vec_flag, fig_title, tick_size)



