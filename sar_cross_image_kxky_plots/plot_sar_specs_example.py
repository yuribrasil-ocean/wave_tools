#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Script: plot_sar_specs_example.py

# Author: Yuri Brasil - yuri.brasil@oceanica.ufrj.br

# Created on Sat Sep  5 15:39:12 2020

# Modification: June 2, 2025

# Objective: Plot SAR cross spectrum (real and imaginary part), exponetial term,
             and wave spectrum using the library sar_and_wave_spectrum_kxky_plot
                
"""


import os
import numpy as np
import scipy.io as sio
from datetime import datetime
from wave_colormaps import wave_spec_cmap, real_part_spec_cmap, imag_part_spec_cmap
from sar_and_wave_spectrum_kxky_plot import wave_spec_kxky_plot, efactor_2D_plot, real_cross_spec_kxky_plot, imag_cross_spec_kxky_plot

# Getting the start time
start_time = datetime.now()

################################# Paths #######################################

# Current path
my_path = os.getcwd()

# Matfiles path
matfiles_path = os.path.join(my_path,'matfiles/')

# Path to save plots
save_path = os.path.join(my_path,'plots/')

########################### Loading the data ##################################

# spec_flag is the flag that defines the type of data
# 'WAVE' = Wave spectrum
# 'INV' = Retrieved wave spectrum
# 'REAL' = Sar Image Wave spectrum
# '2D_EFACTOR' = Efactor matrix

# 'BOTH' = Both, reference and retrieved spectra
# 'INV_DOUBLE' = Two retrieved spectra
# 'REAL_IM' = Both, real and imaginary parts of the cross spectrum

# 'SAR_WAVE' = Four plots, reference, retrieved spectra  real and imaginary parts
# 'SAR_WAVE2' = Four plots, reference, real part of linear, real and imaginary parts of nonlinear 
# 'SAR_WAVE3' = Four plots, reference, real part of linear, real part of nonlinear and retrieved spectrum
# 'SAR_WAVE4' = Four plots, reference, real part of linear, real part of quasi-linear and nonlinear spectrum
# 'EFACTOR' = Efactor plot
# 'EFACTOR_DOUBLE' = Two Efactor plot

spec_flag = 'REAL'

# peak_flag is the flag which allows the peak point to be plotted
# 'ON' = It plots the marker on the peak
peak_flag = True

# norm_flag is the flag which allows the normalization of the whole
# spectrum, ranging between 0 and 1.
norm_flag = False

# Matfile parameters
resolution = 512
# hs = 5
# lp = 380
# dp = 45

hs_list = [4, 2, 2, 2, 4, 3, 3, 3, 4]
dp_list = [45, 45, 0, 90, 45, 45, 90, 0, 0]
lp_list = [500, 500, 200, 200, 200, 300, 300, 300, 250]

######################## Folder and file names  ###############################

# LOOP
for i in range(9):
    hs = hs_list[i]
    lp = lp_list[i]
    dp = dp_list[i]
    
    my_file = f'{matfiles_path}my_matrices_envisat_1001x720_{resolution}pts_jonswap_unimodal_nonlinear_{hs}m_{dp}deg_{lp}m.mat'

###############################################################################

    output = sio.loadmat(my_file)
    # output2 = sio.loadmat(my_file2)
    # output3 = sio.loadmat(my_file3)
    
    # Flag for the mapping
    transf_prefix = 'nonlin_'
    # transf_prefix = 'lin_'
    # transf_prefix = 'qslin_'
    # transf_prefix = 'qs_nonlin_'
    
    param_name = f'{transf_prefix}{hs}m_{lp}m_{dp}' #+ '_partioned'
    # (592 ou 130)
    
    # Kx and Ky vectors, both are equal
    k_vector = output['kx_vec'][0]
    
    # Cut-off wavelength
    cut_off = output['cut_off'][0,0]
    
    # Colorbar title, x and y axis labels
    clb_wave_title = '$\mathbf{m^4}$'
    clb_sar_title = '$\mathbf{m^2}$'
    fig_xlabel = r'$\mathbf{k_y}$ - ' + r'$\mathbf{Range}$ - ' + 'Wavenumber (rad/m)'
    fig_ylabel = r'$\mathbf{k_x}$ - ' + r'$\mathbf{Azimuth}$ - ' + 'Wavenumber \n(rad/m)'
    
    # Ticks and labels sizes
    axis_tick_size = 13
    axis_label_size = 16
    title_size = 14
    colorbar_tick_size = 13
    
    # Function parameters
    theta1 = 360 #330
    theta2 = 180 #150
    axis_lim = 0.08 #0.1
    axis_tick_delta = 0.02
    
    # Axis limits
    if axis_lim == 0.08:
        my_ticks = [' ','-0.06',' ','-0.02',' ','0.02',' ','0.06',' ']
    else:
        my_ticks = [' ','-0.08',' ','-0.04',' ','0.0',' ','0.04',' ','0.08',' ']  
    
    #################################### Plots ####################################
    
    ################################### 1 Plot ####################################
    
    if spec_flag == 'WAVE':
        my_spec = output['spec_kxky']
        maximum_energy = np.max(my_spec)
        fig_title = 'Wave Spectrum'
        file_out = f'{save_path}wave_spec_kxky_{param_name}'
        my_cmap = wave_spec_cmap
    
        wave_spec_kxky_plot(my_spec, k_vector, my_cmap, 
                            my_ticks, axis_tick_delta, axis_lim, theta1, 
                            theta2, cut_off, fig_xlabel, fig_ylabel, 
                            clb_wave_title, fig_title, axis_tick_size, 
                            axis_label_size, colorbar_tick_size, title_size, 
                            file_out, norm_flag, peak_flag)
    
    
    elif spec_flag == '2D_EFACTOR':
        efactor = output['efactor']
        fig_title = '2D Efactor'
        file_out = f'{save_path}efactor_kxky_{param_name}'
        my_cmap = 'viridis'
        
    
        efactor_2D_plot(efactor, k_vector, my_cmap, my_ticks,
                        axis_tick_delta, axis_lim, theta1, theta2, 
                        cut_off, fig_xlabel, fig_ylabel, 
                        fig_title, axis_tick_size, axis_label_size, 
                        colorbar_tick_size, title_size, file_out)   
    
    elif spec_flag == 'REAL':
        my_spec = output['sar_spec']
        maximum_energy = np.max(np.real(my_spec))
        fig_title = 'SAR Cross Spectrum - Real Part'
        file_out = f'{save_path}real_cross_spec_kxky_{param_name}'
        my_cmap = real_part_spec_cmap
        
        real_cross_spec_kxky_plot(my_spec, k_vector, my_cmap, 
                                  my_ticks, axis_tick_delta, axis_lim, theta1, 
                                  theta2, cut_off, fig_xlabel, fig_ylabel, 
                                  clb_sar_title, fig_title, axis_tick_size, 
                                  axis_label_size, colorbar_tick_size, 
                                  title_size, file_out, norm_flag, 
                                  peak_flag)
        
    
    elif spec_flag == 'IMAG':
        my_spec = output['sar_spec']
        maximum_energy = np.max(np.imag(my_spec))
        fig_title = 'SAR Cross Spectrum - Imaginary Part'
        file_out = f'{save_path}imag_cross_spec_kxky_{param_name}'
        my_cmap = imag_part_spec_cmap
        
        
        imag_cross_spec_kxky_plot(my_spec, k_vector, my_cmap, 
                                  my_ticks, axis_tick_delta, axis_lim, theta1, 
                                  theta2, cut_off, fig_xlabel, fig_ylabel, 
                                  clb_sar_title, fig_title, axis_tick_size, 
                                  axis_label_size, colorbar_tick_size, 
                                  title_size, file_out, norm_flag, 
                                  peak_flag)


# Getting the end time and printing running time
end_time = datetime.now()
print(' ')
print('Duration: {}'.format(end_time - start_time))




