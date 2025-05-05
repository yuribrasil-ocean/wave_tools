#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Script: cross_image_spectra_plot_loop_example.py

# Author: Yuri Brasil - yuri.brasil@oceanica.ufrj.br

# Created on Mon May  5 16:10:25 2025

# Modification: May 5, 2025

# Objective: Read and plot a cross image spectrum (f,theta) using several plot 
             functions from cross_image_spectrum_plots.

# Functions:
            
"""

import os
import numpy as np
import scipy.io as sio
from datetime import datetime
import wave_colormaps as wv_cmap
import sar_image_spectrum_plots as sar_plt 

# Getting the start time
start_time = datetime.now()

################################# Paths #######################################

# Current path
my_path = os.getcwd()

# Path to save plots
save_path = os.path.join(my_path,'plots/')

# List of prefixes
prefixes_list = ['REAL','IMAG','COMPLEX']

# Lists of keys
spec_list = ['spec_f_real', 'spec_k_imag', 'spec2d']
freq_list = ['real_freq_vec','imag_k_vec','freq_vec']
dir_list = ['image_dir_vec','image_dir_vec','dir_vec']

# Images list
images_list = ['9','56','72']

# List of resolutions
res_list = ['36x24','36x24','24x25']

# Title lists
title_lists = ['Real cross spectrum density', 'Imaginary cross spectrum density', 'Real cross spectrum density']

for n in images_list:
    for i in range(3):
        
        print(f'Cross Spectra - {prefixes_list[i]}\n')  
    
        ######################## Extracting data ##############################
    
        # Reading the file
        my_file = os.path.join(my_path,f'data/{prefixes_list[i]}_{res_list[i]}_image_{n}.mat')
    
        # .mat file
        output = sio.loadmat(my_file)
    
        ############### Real, imaginary and complex spectra ###################
    
        # Cross image spectrum and vectors
        spec2d = output[spec_list[i]]
        
        if prefixes_list[i] == 'IMAG':
            k_vec = output[freq_list[i]][0]
            freq_vec = np.sqrt(k_vec*9.80665)/(2*np.pi)    
            # spec2d = spec2d[:,::-1]
            
            # a = 2
    
            
            # CALCULAR DFDK
            # jacobian = np.sqrt(9.80665/k_vec)
            # spec2d = np.divide(spec2d,jacobian)            
        else:
            freq_vec = output[freq_list[i]][0]
        
        freq_vec = output[freq_list[i]][0]
        dir_vec = output[dir_list[i]][0]
    
        ######################### Plot parameters #############################
    
        # Param dictionary
        param_dict = {}
        
        # Flag for the normalization 
        norm_flag = False
    
        # Flag for mark the spectrum peak
        peak_flag = True
    
        # Flag to plot the wave parameters in a text box
        param_flag = False    
        
        # Setting figure title
        fig_title = title_lists[i]
        
        # Tick size
        tick_size = 1
        
        # Sets the filename    
        fname = f'{save_path}/{prefixes_list[i]}_cartesian_{prefixes_list[i]}_cross_spec_freq_image_{n}'
    
        ##################### Wave spectra plots ##############################
    
        if prefixes_list[i] == 'REAL':
            sar_plt.real_cross_cartesian_spec(dir_vec, freq_vec, spec2d, param_dict, 
                                              fname, wv_cmap.real_part_spec_cmap, 
                                              norm_flag, peak_flag, param_flag, 
                                              'freq', fig_title, tick_size)
            
        elif prefixes_list[i] == 'IMAG':
            sar_plt.imag_cross_cartesian_spec(dir_vec, freq_vec, spec2d, param_dict, 
                                              fname, wv_cmap.imag_part_spec_cmap, 
                                              norm_flag, peak_flag, param_flag, 
                                              'freq', fig_title, tick_size)
            
        elif prefixes_list[i] == 'COMPLEX':
            sar_plt.real_cross_cartesian_spec(dir_vec, freq_vec, spec2d, param_dict, 
                                              fname, wv_cmap.real_part_spec_cmap, 
                                              norm_flag, peak_flag, param_flag, 
                                              'freq', fig_title, tick_size)


# Getting the end time and printing running time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))




