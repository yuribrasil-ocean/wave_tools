#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Script: plot_sentinel1_L2-ocn_OSW_wave_and_cross_spectra.py

# Author: Yuri Brasil - yuri.brasil@oceanica.ufrj.br

# Created on Mon Jun  9 18:53:23 2025

# Modification: July 23, 2025

# Objective: Plot Sentinel or Envisat image cross spectra

# Functions: 
            
"""

import os
import numpy as np
import scipy.io as sio
from datetime import datetime
import wave_colormaps as wv_colors
import wave_spectral_parameters as par
import satellite_spectrum_plots as sat


# Getting the start time
start_time = datetime.now()

################################# Paths #######################################

# main_path = '/run/media/numa20'
# main_path = '/media/yuri'

# Main path
main_path = os.getcwd()

# Track number
track_number = '021944'

# Matfile path
matfile_path = os.path.join(main_path,'matfiles')

# Save path
save_path = os.path.join(main_path,'plots3')

# Sentinel matfile
sentinel_mat = sio.loadmat(f'{matfile_path}/sentinel1_osw_track_{track_number}_data.mat')

# Sentinel Quasilinear results
sentinel_qsl_mat = sio.loadmat(f'{matfile_path}/quasilinear_data_sentinel_track_{track_number}_wvi_cutoff_efactor_01_all_filters_OFF_2time.mat')
                                                
# Sentinel OSW data
sentinel_osw_mat = sio.loadmat(f'{matfile_path}/sentinel1_osw_track_{track_number}_retrieved_spectra.mat')

# Sentinel track vectors
im_vec = sentinel_mat['image_n'][0]
freq_vec = sentinel_osw_mat['f'][0]
dir_vec = sentinel_mat['dire'][0]
df = par.d_logarithmic(freq_vec)

# Sentinel track (θ,f) spectra
wave_spec_ft = sentinel_osw_mat['polar_spec_f_theta']
real_spec_ft = sentinel_mat['spec_f_real']
imag_spec_ft = sentinel_mat['spec_f_imag']
complex_spec_ft = real_spec_ft + 1j * imag_spec_ft

# Sentinel quasilinear data (kx,ky) spectra

# Flatten the Hs_list
Hs_list_flat = sentinel_qsl_mat['hs_list'].flatten() 

# Get the indices where Hs_list is not zero
valid_indices = np.arange(5)

# Select only the valid slices from the 3D array
cross_spec_kxky_before = sentinel_qsl_mat['cross_spec_kxky_before_all'][:, :, valid_indices]
cross_spec_kxky_after = sentinel_qsl_mat['cross_spec_kxky_after_all'][:, :, valid_indices]
cutoff_mpi = sentinel_qsl_mat['cutoff_mpi_list'][valid_indices]
cutoff_wvi = sentinel_qsl_mat['cutoff_wvi_list'][valid_indices]
efactor_mod = sentinel_qsl_mat['efactor_mod_all'][:, :, valid_indices]
lin_den = sentinel_qsl_mat['lin_den_all'][:, :, valid_indices]
qsl_spec_kxky = sentinel_qsl_mat['qsl_spec_kxky_all'][:, :, valid_indices]
n_image = sentinel_qsl_mat['n_image_list'][valid_indices] 

# Cut-off resolution
cutoff_res = sentinel_osw_mat['res_by_direction_vec']

# Gravity acceleration
g = 9.80665

# OSW wavenumber,wavelength and frequency vectors
# L_vec = (2*np.pi)/cutoff_res
cutoff_res_freq = np.sqrt(g/(cutoff_res*2*np.pi))

# Cutoff list
cutoff = sentinel_mat['az_cutoff'][0]
cutoff_freq = np.sqrt(g/(cutoff*2*np.pi))

# Track angle
track_angle = sentinel_mat['ground_track'][0]

for n in range(5):
    
    ############################### E(θ,f) ####################################
    
    print(f'Wave Spectrum (θ,f) from image {n+1}\n')

    # Spectral parameters
    hs = par.hs_spec(dir_vec, freq_vec, df, wave_spec_ft[:,:,n], True)
    fp, tp, lp = par.peak_frequency(dir_vec, freq_vec, wave_spec_ft[:,:,n], True)
    dp = par.peak_direction(dir_vec, freq_vec, df, wave_spec_ft[:,:,n])
    
    # Param dictionary
    param_dict = {'Hs':hs,'Fp':fp,'Tp':tp,'Lp':lp,'Dp':dp}
    
    # Flag for the normalization 
    norm_flag = False

    # Flag for mark the spectrum peak
    peak_flag = True

    # Flag to plot the wave parameters in a text box
    param_flag = True    
    
    # Setting figure title
    fig_title = f'OSW Wave Spectrum Density - Image {n+1}'
    
    # Sets the filename    
    fname = f'{save_path}/sentinel_track_{track_number}_n_image_{n+1}_OSW_cartesian_wave_spec_freq'
    
    # Flag for frequency or piod
    vec_flag = 'freq'
    
    # Size of ticks
    tick_size = 2

    sat.cartesian_spec(dir_vec, freq_vec, wave_spec_ft[:,:,n], param_dict, 
                       fname, wv_colors.wave_spec_cmap, norm_flag, peak_flag, 
                       param_flag, vec_flag, cutoff[n], cutoff_res_freq[:,n], 
                       cutoff_freq[n], track_angle[n], fig_title, tick_size)
    
    print(f'Real part of Image Cross Spectrum (θ,f) from image {n+1}\n')
    
    # Setting figure title
    fig_title_real = f'OSW Real part of Image Cross Spectrum - Image {n+1}'
    
    # Sets the filename    
    fname_real = f'{save_path}/sentinel_track_{track_number}_n_image_{n+1}_OSW_real_cross_spec_freq'
    
    # Flag to plot the wave parameters in a text box
    param_flag2 = False  
    
    # Flag for mark the spectrum peak
    peak_flag2 = True

    sat.real_cartesian_spec(dir_vec, freq_vec, complex_spec_ft[:,:,n], param_dict, 
                            fname_real, wv_colors.real_part_spec_cmap, norm_flag, 
                            peak_flag2, param_flag2, vec_flag, cutoff[n], 
                            cutoff_res_freq[:,n], cutoff_freq[n], 
                            track_angle[n], fig_title_real, tick_size)
    
    print(f'Imaginary part of Image Cross Spectrum (θ,f) from image {n+1}\n')

    # Setting figure title
    fig_title_imag = f'OSW Imaginary part of Image Cross Spectrum - Image {n+1}'
    
    # Sets the filename    
    fname_imag = f'{save_path}/sentinel_track_{track_number}_n_image_{n+1}_OSW_imag_cross_spec_freq'

    sat.imag_cartesian_spec(dir_vec, freq_vec, complex_spec_ft[:,:,n], param_dict, 
                            fname_imag, wv_colors.imag_part_spec_cmap, norm_flag, 
                            peak_flag2, param_flag2, vec_flag, cutoff[n], 
                            cutoff_res_freq[:,n], cutoff_freq[n], 
                            track_angle[n], fig_title_imag, tick_size)
    
    ############################## E(kx,ky) ###################################

    # Data
    my_cross_spec_before = cross_spec_kxky_before[:,:,n]
    my_cross_spec_after = cross_spec_kxky_after[:,:,n]
    my_cutoff_mpi = cutoff_mpi[n,0]
    my_cutoff_wvi = cutoff_wvi[n,0]
    
    param_name = f'sentinel_track_{track_number}_n_image_{n+1}'
    
    # Kx and Ky vectors, both are equal
    k_vector = sentinel_qsl_mat['kxky_vec'][:,0]
    
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
    axis_lim = 0.08 # 0.1 0.08
    # axis_tick_delta = 0.04 #0.02 0.02
    
    # Axis limits
    if axis_lim == 0.08:
        axis_tick_delta = 0.02
        my_ticks = [' ','-0.06',' ','-0.02',' ','0.02',' ','0.06',' ']
    elif axis_lim == 0.1:
        axis_tick_delta = 0.02
        my_ticks = [' ','-0.08',' ','-0.04',' ','0.0',' ','0.04',' ','0.08',' ']
    elif axis_lim == 0.2:
        axis_tick_delta = 0.04
        my_ticks = [' ','-0.16',' ','-0.08',' ','0.0',' ','0.08',' ','0.16',' ']
    
    ################################## Plots ##################################


    print(f'Wave Spectrum (kx,ky) from image {n+1}\n')
    
    fig_title = 'Wave Spectrum'
    file_out = f'{save_path}/qsl_wave_spec_kxky_{param_name}'

    sat.wave_spec_kxky_plot(qsl_spec_kxky[:, :, n], k_vector, 
                            wv_colors.wave_spec_cmap, my_ticks, 
                            axis_tick_delta, axis_lim, theta1, theta2, 
                            my_cutoff_wvi, my_cutoff_mpi, fig_xlabel, 
                            fig_ylabel, clb_wave_title, fig_title, 
                            axis_tick_size, axis_label_size, 
                            colorbar_tick_size, title_size, file_out, 
                            norm_flag=False, peak_flag=True)
    
    
    
    print(f'Real part of Image Cross Spectrum (kx,ky) BEFORE from image {n+1}\n')
    

    fig_title = 'SAR Cross Spectrum - Real Part'
    file_out = f'{save_path}/real_cross_spec_kxky_{param_name}_before'
           
    sat.real_cross_spec_kxky_plot(my_cross_spec_before, k_vector, 
                                  wv_colors.real_part_spec_cmap, my_ticks, 
                                  axis_tick_delta, axis_lim, theta1, theta2, 
                                  my_cutoff_wvi, my_cutoff_mpi, fig_xlabel, 
                                  fig_ylabel, clb_sar_title,  fig_title, 
                                  axis_tick_size, axis_label_size, 
                                  colorbar_tick_size, title_size, file_out, 
                                  norm_flag=False, peak_flag=True)
    
    print(f'Real part of Image Cross Spectrum (kx,ky) AFTER from image {n+1}\n')
    
    fig_title = 'SAR Cross Spectrum - Real Part'
    file_out = f'{save_path}/real_cross_spec_kxky_{param_name}_after'
        
    sat.real_cross_spec_kxky_plot(my_cross_spec_after, k_vector, 
                                  wv_colors.real_part_spec_cmap, my_ticks, 
                                  axis_tick_delta, axis_lim, theta1, theta2, 
                                  my_cutoff_wvi, my_cutoff_mpi, fig_xlabel, 
                                  fig_ylabel, clb_sar_title,  fig_title, 
                                  axis_tick_size, axis_label_size, 
                                  colorbar_tick_size, title_size, file_out, 
                                  norm_flag=False, peak_flag=True)
    
    print(f'Imaginary part of Image Cross Spectrum (kx,ky) BEFORE from image {n+1}\n')
    

    fig_title = 'SAR Cross Spectrum - Imaginary Part'
    file_out = f'{save_path}/imag_cross_spec_kxky_{param_name}_before'  
    
    sat.imag_cross_spec_kxky_plot(my_cross_spec_before, k_vector, 
                                  wv_colors.imag_part_spec_cmap, my_ticks, 
                                  axis_tick_delta, axis_lim, theta1, theta2, 
                                  my_cutoff_wvi, my_cutoff_mpi, fig_xlabel, 
                                  fig_ylabel, clb_sar_title, fig_title, 
                                  axis_tick_size, axis_label_size, 
                                  colorbar_tick_size, title_size, file_out, 
                                  norm_flag=False, peak_flag=True) 
    
    print(f'Imaginary part of Image Cross Spectrum (kx,ky) AFTER from image {n+1}\n')

    fig_title = 'SAR Cross Spectrum - Imaginary Part'
    file_out = f'{save_path}/imag_cross_spec_kxky_{param_name}_after'  
    
    sat.imag_cross_spec_kxky_plot(my_cross_spec_after, k_vector, 
                                  wv_colors.imag_part_spec_cmap, my_ticks, 
                                  axis_tick_delta, axis_lim, theta1, theta2, 
                                  my_cutoff_wvi, my_cutoff_mpi, fig_xlabel, 
                                  fig_ylabel, clb_sar_title, fig_title, 
                                  axis_tick_size, axis_label_size, 
                                  colorbar_tick_size, title_size, file_out, 
                                  norm_flag=False, peak_flag=True) 
    
    
    
    
    