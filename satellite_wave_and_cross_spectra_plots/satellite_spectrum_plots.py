#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Saturday September 5th 2020

# Script: satellite_spectrum_plots

# Author: Yuri Brasil

# e-mail: yuri.brasil@oceanica.ufrj.br

# Modification: July 6, 2025

# Objective: Plot different forms of wave density spectrum (f,theta)

Functions:
    
    omnidirectional_spec - Function to plot the 1D wave spectrum
    
    polar_spec - Function to plot the 2D polar contourf graph of the wave
                 spectrum.
                 
    cartesian_spec - Function to plot the 2D contourf graph of the wave
                     spectrum. 
                     
    map_spec - Function to plot the 2D contourf graph of the wave spectrum
               as a heat map, showing the values on the graph.

"""

import math 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

########################## Setting Figure parameters #########################

# Figure layout and quality
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.dpi'] = 300

# Don't cut nothing in the figure saved
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'

########################## Spectrum plot functions ############################

############### Correcting spectrum orientation (if needed) ###################

def spec_orientation(dir_vec, freq_vec, spec):
    
    # Checking the spectrum orientation and, if necessary, correcting 
    # the spectrum orientation that should be E(θ,f) or E(θ,k)
    if spec.shape == (len(dir_vec),len(freq_vec)):
        spec2d = spec
    elif spec.shape == (len(freq_vec),len(dir_vec)):
        spec2d = np.transpose(spec) 
    else:
        raise ValueError("Input spectrum dimensions do not match frequency and direction vectors.")
    
    return spec2d


################################# E(θ,f) ######################################




##################### Cartesian Wave Spectrum Plot ############################  

def cartesian_spec(dir_vec, freq_vec, spec2d, param_dict, fname, my_cmap, 
                   norm_flag, peak_flag, param_flag, vec_flag, cutoff, 
                   cutoff_res_freq, cutoff_freq, track_angle, fig_title, 
                   tick_size): 

############################ Adjusting parameters #############################
    
    # Figure size and quality
    mpl.rcParams['figure.figsize'] = (8, 5)
    
    # Checking the spectrum orientation
    spec2d = spec_orientation(dir_vec, freq_vec, spec2d)    
    
    # Editing the spectrum and direction vector to fill the trigonometric 
    # circle
    spec2d_full = np.vstack((spec2d,spec2d[0,:]))
    dir_vec_full = np.hstack((dir_vec,360)) 

    # Normalize the spectrum
    if norm_flag == True:
        spec2d_full = spec2d_full/np.max(spec2d_full)
        
###################### Creating ticks and tick labels #########################
    
    # Create the dir_ticks
    dir_ticks = np.arange(0,360+15,15)

    dir_tick_labels = [' ']*len(dir_ticks)
    for d in range(2,len(dir_ticks),2):
        dir_tick_labels[d] = f'{int(dir_ticks[d])}°'    
    
    # Preparing the vec_ticks
    freq_ticks = np.arange(0.0,0.5,0.05)    

    # Frequency tick labels    
    freq_tick_labels = [f'{np.round(i, 2)}Hz' for i in freq_ticks]     
    
    # Period tick labels    
    period_tick_labels = ['inf' if i == 0 else f'{np.round(1/i, 1)}s' for i in freq_ticks]

###############################################################################

    # Creating the ticks of period vector
    if vec_flag == 'per':
        x_tick_labels = period_tick_labels
        x_label = 'Period'
    elif vec_flag == 'freq':
        x_tick_labels = freq_tick_labels
        x_label = 'Frequency'
        
    # Setting up the maximum value for ticks and levels
    if np.max(spec2d_full) < 10:
        my_max = np.round(np.max(spec2d_full),1)
    else:
        my_max = int(np.round(np.max(spec2d_full)))
    
    # Set my_max = 1 if there is no energy
    if my_max < 1.0:
        my_max = 1

    # Define step based on my_max
    step_lookup = {1: 0.05, 2: 0.1, 5: 0.2, 10: 0.5, 20: 1, 30: 2, 
                   50: 5, 100: 10, 200: 20, 500: 50, 1000: 100, 
                   2000: 200, 10000: 1000}
    step = next((v for k, v in step_lookup.items() if my_max <= k), 2000)
    
    # Creating the ticks and tick labels
    cbar_ticks = np.arange(step,my_max + step, step)
    cbar_ticks = np.round(cbar_ticks, 1) if step < 1 else cbar_ticks
    cbar_label = []    
        
    for i in range(len(cbar_ticks)):
        if (i % 2) == 0:
            cbar_label.append(str(cbar_ticks[i]))
        else:
            cbar_label.append(' ') 
        
    # The step for the contourf plot    
    cntrf_step = step/2  
    first_level = cntrf_step

################################## The Plot ###################################

    # figure 1
    fig, ax = plt.subplots()
       
    # Creating the levels
    contourf_levels = np.arange(cntrf_step, cbar_ticks[-1]+cntrf_step, cntrf_step)
    contourf_levels = np.round(contourf_levels,2)
    line_levels = np.arange(step, my_max+step, step)   
    
    # Plotting the lines
    plt.contour(freq_vec, dir_vec_full, spec2d_full, colors='black', 
                levels=[first_level], alpha=0.8, linewidths=0.2, zorder=20)
    
    plt.contour(freq_vec, dir_vec_full, spec2d_full, colors='black', 
                levels=line_levels, alpha=0.8, linewidths=0.2, zorder=20)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(freq_vec, dir_vec_full, spec2d_full, cmap=my_cmap, 
                       levels=contourf_levels, alpha=0.95, zorder=15)
    
    # Plotting the maximum point
    if peak_flag == True:
        [max_y, max_x] = np.where(spec2d_full == np.max(spec2d_full))
        plt.scatter(freq_vec[max_x[0]], dir_vec_full[max_y[0]], s=40, 
                    color='white', marker='x', linewidths=1.2, zorder=30) 
        
    # Plotting the parameters box
    if param_flag == True:
        plt.figtext(0.71, 0.78,                         
                    'Hs = ' + str(np.round(param_dict['Hs'],2)) + 'm' + '\n'
                    'Fp = ' + str(np.round(param_dict['Fp'],2)) + 'Hz' + '\n'
                    'Tp = ' + str(np.round(param_dict['Tp'],2)) + 's' + '\n'
                    'Lp = ' + str(np.round(param_dict['Lp'],2)) + 'm' + '\n'
                    'Dp = ' + str(np.round(param_dict['Dp'],2)) + '°'
                    ,fontsize=8, color='black', bbox=dict(facecolor='w', alpha=1))       
        
    # Cutoff plot
    cutoff_res_freq = np.concatenate([[cutoff_res_freq[-1]], cutoff_res_freq])
    plt.plot(cutoff_res_freq, dir_vec_full, linestyle='--', color='red', 
             alpha=0.9, linewidth=1, zorder=30)
    plt.text(cutoff_freq + 0.002, track_angle, str(int(cutoff)) + 'm', 
             fontsize=10, color='red', fontweight='bold', zorder=30)
            
    # Setting the ticks and ticklabels
    ax.set_xticks(freq_ticks)  
    ax.set_yticks(dir_ticks)  
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=10)
    ax.set_yticklabels(dir_tick_labels, rotation=0, fontsize=10)
    
    # Setting the parameters of labels
    plt.xlim([0, 0.5])    
    plt.tick_params(length=tick_size)
    
############################# Colorbar and title ##############################    
    
    # Setting the colorbar
    cbar = fig.colorbar(cnt1, ticks=cbar_ticks, pad=0.02)
    
    # Stablish the colorbar tick labels to be equal to the ticks
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
    
    # Adjust the colorbar ticks size    
    cbar.ax.tick_params(labelsize=10)
        
    if vec_flag == 'freq':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} Hz^{-1} \hspace{0.2} degree^{-1}})$'
    elif vec_flag == 'per':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} s \hspace{0.2} degree^{-1}})$'
    
    if norm_flag == True:
        title_unity = '(normalized)'

######################### Title and labels #################################### 

    # Adding grid
    ax.grid(which='major', color='black', alpha=0.2, linestyle='dotted')

    # Fig labels        
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel('Direction', fontsize=14, fontweight='bold')
    
    # Showing and saving the plot
    my_title = fig_title + ' ' + title_unity
    plt.title(my_title, fontsize=10, fontweight='bold', y=1.0)

    # Showing and saving the plot
    plt.show()
    plt.savefig(fname, pad_inches=0.1)
    plt.close()

############# Cartesian Real part of Image Cross Spectrum Plot ################  

def real_cartesian_spec(dir_vec, freq_vec, my_spec, param_dict, fname, my_cmap, 
                        norm_flag, peak_flag, param_flag, vec_flag, cutoff, 
                        cutoff_res_freq, cutoff_freq, track_angle, fig_title,
                        tick_size): 

############################ Adjusting parameters #############################
    
    # Figure size and quality
    mpl.rcParams['figure.figsize'] = (8, 5)
    
    # The spectrum
    real_spec = np.real(my_spec)
    imag_spec = np.imag(my_spec)
    
    # Checking the spectrum orientation
    real_spec = spec_orientation(dir_vec, freq_vec, real_spec)    
    
    # Editing the spectrum and direction vector to fill the trigonometric 
    # circle
    real_spec_full = np.vstack((real_spec,real_spec[0,:]))
    dir_vec_full = np.hstack((dir_vec,360)) 
    
############## Peak value without 180° directional ambiguity ################## 

    # Set a small tolerance to catch small numerical differences
    tol = 1e-5
    
    # Step 1: Find all real peaks (within tolerance)
    real_max_energy = np.max(real_spec)
    real_max_coords = np.column_stack(np.where(np.abs(real_spec - real_max_energy) <= tol))

    # Step 2: Evaluate 3x3 neighborhood means in the imaginary part
    def get_window_mean(matrix, y, x):
        y_min = max(y - 1, 0)
        y_max = min(y + 2, matrix.shape[0])
        x_min = max(x - 1, 0)
        x_max = min(x + 2, matrix.shape[1])
        window = matrix[y_min:y_max, x_min:x_max]
        return np.mean(window)
    
    # Step 3: Loop and select the best peak
    best_coord = None
    best_mean = -np.inf  # We want the most positive region
    
    for coord in real_max_coords:
        y, x = coord
        mean_val = get_window_mean(imag_spec, y, x)
        if mean_val > best_mean and mean_val > 0:
            best_mean = mean_val
            best_coord = coord
    
    # Step 4: Fallback if no positive mean region is found
    if best_coord is None:
        print("Warning: No real peak corresponds to a positive region in the imaginary part.")
        best_coord = real_max_coords[0]  # fallback to the first max
    
    # Final selected peak coordinates
    max_y, max_x = best_coord

    # Normalize the spectrum
    if norm_flag == True:
        real_spec_full = real_spec_full/np.max(real_spec_full)
        
###################### Creating ticks and tick labels #########################
    
    # Create the dir_ticks
    dir_ticks = np.arange(0,360+15,15)

    dir_tick_labels = [' ']*len(dir_ticks)
    for d in range(2,len(dir_ticks),2):
        dir_tick_labels[d] = f'{int(dir_ticks[d])}°'    
    
    # Preparing the vec_ticks
    freq_ticks = np.arange(0.0,0.5,0.05)    

    # Frequency tick labels    
    freq_tick_labels = [f'{np.round(i, 2)}Hz' for i in freq_ticks]     
    
    # Period tick labels    
    period_tick_labels = ['inf' if i == 0 else f'{np.round(1/i, 1)}s' for i in freq_ticks]

###############################################################################

    # Creating the ticks of period vector
    if vec_flag == 'per':
        x_tick_labels = period_tick_labels
        x_label = 'Period'
    elif vec_flag == 'freq':
        x_tick_labels = freq_tick_labels
        x_label = 'Frequency'
        
    # Setting up the maximum value for ticks and levels
    if np.max(real_spec_full) < 10:
        my_max = np.round(np.max(real_spec_full),1)
    else:
        my_max = int(np.round(np.max(real_spec_full)))
    
    # Set my_max = 1 if there is no energy
    if my_max < 1.0:
        my_max = 1

    # Define step based on my_max
    step_lookup = {1: 0.05, 2: 0.1, 5: 0.2, 10: 0.5, 20: 1, 30: 2, 
                   50: 5, 100: 10, 200: 20, 500: 50, 1000: 100, 
                   2000: 200, 10000: 1000}
    step = next((v for k, v in step_lookup.items() if my_max <= k), 2000)
    
    # Creating the ticks and tick labels
    cbar_ticks = np.arange(step,my_max + step, step)
    cbar_ticks = np.round(cbar_ticks, 1) if step < 1 else cbar_ticks
    cbar_label = []    
        
    for i in range(len(cbar_ticks)):
        if (i % 2) == 0:
            cbar_label.append(str(cbar_ticks[i]))
        else:
            cbar_label.append(' ') 
            
            
        
    # The step for the contourf plot    
    cntrf_step = step/2  
    first_level = cntrf_step

################################## The Plot ###################################

    # figure 1
    fig, ax = plt.subplots()
       
    # Creating the levels
    contourf_levels = np.arange(cntrf_step, cbar_ticks[-1]+cntrf_step, cntrf_step)
    contourf_levels = np.round(contourf_levels,2)
    line_levels = np.arange(step, my_max+step, step)   
    
    # Plotting the lines
    plt.contour(freq_vec, dir_vec_full, real_spec_full, colors='black', 
                levels=[first_level], alpha=0.8, linewidths=0.2, zorder=20)
    
    plt.contour(freq_vec, dir_vec_full, real_spec_full, colors='black', 
                levels=line_levels, alpha=0.8, linewidths=0.2, zorder=20)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(freq_vec, dir_vec_full, real_spec_full, cmap=my_cmap, 
                       levels=contourf_levels, alpha=0.95, zorder=15)
    
    # Plotting the maximum point
    if peak_flag == True:
        plt.scatter(freq_vec[max_x], dir_vec_full[max_y], s=40, color='white',
                    edgecolors='black', marker='x', linewidths=1.2, zorder=30) 
        
    # Plotting the parameters box
    if param_flag == True:
        plt.figtext(0.71, 0.78,                         
                    'Hs = ' + str(np.round(param_dict['Hs'],2)) + 'm' + '\n'
                    'Fp = ' + str(np.round(param_dict['Fp'],2)) + 'Hz' + '\n'
                    'Tp = ' + str(np.round(param_dict['Tp'],2)) + 's' + '\n'
                    'Lp = ' + str(np.round(param_dict['Lp'],2)) + 'm' + '\n'
                    'Dp = ' + str(np.round(param_dict['Dp'],2)) + '°'
                    ,fontsize=8, color='black', bbox=dict(facecolor='w', alpha=1)) 
        
    # Cutoff plot
    cutoff_res_freq = np.concatenate([[cutoff_res_freq[-1]], cutoff_res_freq])
    plt.plot(cutoff_res_freq, dir_vec_full, linestyle='--', color='red', 
             alpha=0.9, linewidth=1, zorder=30)
    plt.text(cutoff_freq + 0.002, track_angle, str(int(cutoff)) + 'm', 
             fontsize=10, color='red', fontweight='bold', zorder=30)   
    
    # Setting the ticks and ticklabels
    ax.set_xticks(freq_ticks)  
    ax.set_yticks(dir_ticks)  
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=10)
    ax.set_yticklabels(dir_tick_labels, rotation=0, fontsize=10)
    
    # Setting the parameters of labels
    plt.xlim([0, 0.5])    
    plt.tick_params(length=tick_size)
    
############################# Colorbar and title ##############################    
    
    # Setting the colorbar
    cbar = fig.colorbar(cnt1, ticks=cbar_ticks, pad=0.02)
    
    # Stablish the colorbar tick labels to be equal to the ticks
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
    
    # Adjust the colorbar ticks size    
    cbar.ax.tick_params(labelsize=10)
           
    if vec_flag == 'freq':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} Hz^{-1} \hspace{0.2} degree^{-1}})$'
    elif vec_flag == 'per':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} s \hspace{0.2} degree^{-1}})$'
    
    if norm_flag == True:
        title_unity = '(normalized)'

######################### Title and labels #################################### 

    # Adding grid
    ax.grid(which='major', color='black', alpha=0.2, linestyle='dotted')

    # Fig labels        
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel('Direction', fontsize=14, fontweight='bold')
    
    # Showing and saving the plot
    my_title = fig_title + ' ' + title_unity
    plt.title(my_title, fontsize=10, fontweight='bold', y=1.0)

    # Showing and saving the plot
    plt.show()
    plt.savefig(fname, pad_inches=0.1)
    plt.close()
    
    
########## Cartesian Imaginary part of Image Cross Spectrum Plot ##############  

def imag_cartesian_spec(dir_vec, freq_vec, my_spec, param_dict, fname, my_cmap, 
                        norm_flag, peak_flag, param_flag, vec_flag, cutoff, 
                        cutoff_res_freq, cutoff_freq, track_angle, fig_title,
                        tick_size): 

############################ Adjusting parameters #############################
    
    # Figure size and quality
    mpl.rcParams['figure.figsize'] = (8, 5)
    
    # The spectrum
    real_spec = np.real(my_spec)
    imag_spec = np.imag(my_spec)
    
    # Checking the spectrum orientation
    imag_spec = spec_orientation(dir_vec, freq_vec, imag_spec)    
    
    # Editing the spectrum and direction vector to fill the trigonometric 
    # circle
    imag_spec_full = np.vstack((imag_spec,imag_spec[0,:]))
    dir_vec_full = np.hstack((dir_vec,360)) 

    # Normalize the spectrum
    if norm_flag == True:
        imag_spec_full = imag_spec_full/np.max(imag_spec_full)
        
############## Peak value without 180° directional ambiguity ################## 
   
    # Set a small tolerance to catch small numerical differences
    tol = 1e-5
    
    # Step 1: Find all real peaks (within tolerance)
    real_max_energy = np.max(real_spec)
    real_max_coords = np.column_stack(np.where(np.abs(real_spec - real_max_energy) <= tol))

    # Step 2: Evaluate 3x3 neighborhood means in the imaginary part
    def get_window_mean(matrix, y, x):
        y_min = max(y - 1, 0)
        y_max = min(y + 2, matrix.shape[0])
        x_min = max(x - 1, 0)
        x_max = min(x + 2, matrix.shape[1])
        window = matrix[y_min:y_max, x_min:x_max]
        return np.mean(window)
    
    # Step 3: Loop and select the best peak
    best_coord = None
    best_mean = -np.inf  # We want the most positive region
    
    for coord in real_max_coords:
        y, x = coord
        mean_val = get_window_mean(imag_spec, y, x)
        if mean_val > best_mean and mean_val > 0:
            best_mean = mean_val
            best_coord = coord
    
    # Step 4: Fallback if no positive mean region is found
    if best_coord is None:
        print("Warning: No real peak corresponds to a positive region in the imaginary part.")
        best_coord = real_max_coords[0]  # fallback to the first max
    
    # Final selected peak coordinates
    max_y, max_x = best_coord        
        
###################### Creating ticks and tick labels #########################
    
    # Create the dir_ticks
    dir_ticks = np.arange(0,360+15,15)

    dir_tick_labels = [' ']*len(dir_ticks)
    for d in range(2,len(dir_ticks),2):
        dir_tick_labels[d] = f'{int(dir_ticks[d])}°'    
    
    # Preparing the vec_ticks
    freq_ticks = np.arange(0.0,0.5,0.05)    

    # Frequency tick labels    
    freq_tick_labels = [f'{np.round(i, 2)}Hz' for i in freq_ticks]     
    
    # Period tick labels    
    period_tick_labels = ['inf' if i == 0 else f'{np.round(1/i, 1)}s' for i in freq_ticks]

###############################################################################

    # Creating the ticks of period vector
    if vec_flag == 'per':
        x_tick_labels = period_tick_labels
        x_label = 'Period'
    elif vec_flag == 'freq':
        x_tick_labels = freq_tick_labels
        x_label = 'Frequency'
        
################################### levels #################################### 
    
    # Normalizing the spectrum
    if norm_flag == True:
        imag_spec = imag_spec/np.max(imag_spec)    
           
    # Set the maximum value for ticks and levels
    if np.max(imag_spec) < 10:
        my_max = np.ceil(np.max(imag_spec))
    else:
        # my_max = int(np.round(np.max(imag_spec)))    
        my_max = int(np.ceil(np.max(imag_spec)))
        
    # Set my_max = 1 if there is no energy
    if my_max < 1.0:
        my_max = 1
        
    # Set the minimum energy value
    my_min = -my_max
    
    #Lookup table for tick step
    step_lookup = {1: 0.2, 2: 0.4, 5: 1, 10: 2, 20: 4, 30: 5, 50: 10, 100: 20, 
                   200: 40, 500: 100, 1000: 200, 2000: 400, 10000: 2000}
    
    # Set tick_step, cntrf_step and cntr_step
    step = next((v for k, v in step_lookup.items() if my_max <= k), 4000)
    
    # Creating the ticks and tick labels
    cbar_ticks_pos = np.arange(step, my_max + step/2, step)
    cbar_ticks_neg = -1*cbar_ticks_pos[::-1]
    cbar_ticks = np.concatenate((cbar_ticks_neg, [0], cbar_ticks_pos))
    cbar_ticks = np.round(cbar_ticks, 1) if step < 1 else cbar_ticks
    cbar_label = []   
        
    for i in range(len(cbar_ticks)):
        if (i % 2) == 0:
            cbar_label.append(str(cbar_ticks[i]))
        else:
            cbar_label.append(' ') 

    # Set the minimum energy value
    my_min = -my_max

################################## The Plot ###################################

    # figure 1
    fig, ax = plt.subplots()
       
    # Creating steps
    contourf_step = step/4  
    cntr_step = step/2     
    
    # Create contourf and contour levels
    contourf_levels_pos = np.arange(contourf_step, my_max + contourf_step/2, contourf_step)
    contourf_levels_neg = -1*contourf_levels_pos[::-1]
    contourf_levels = np.concatenate((contourf_levels_neg, [0], contourf_levels_pos))
    
    cntr_levels_pos = np.arange(cntr_step, my_max + cntr_step/2, cntr_step)
    cntr_levels_neg = -1*cntr_levels_pos[::-1]
    cntr_levels = np.concatenate((cntr_levels_neg, [0], cntr_levels_pos))    
    
    # Plotting the lines
    plt.contour(freq_vec, dir_vec_full, imag_spec_full, colors='black', 
                levels=[cntr_step], alpha=0.8, linewidths=0.2,
                vmin=my_min, vmax=my_max, zorder=4)
    
    plt.contour(freq_vec, dir_vec_full, imag_spec_full, colors='black', 
                levels=cntr_levels, alpha=0.8, linewidths=0.2, 
                vmin=my_min, vmax=my_max, zorder=4)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(freq_vec, dir_vec_full, imag_spec_full, cmap=my_cmap, 
                       levels=contourf_levels, alpha=0.8, vmin=cbar_ticks[0], 
                       vmax=cbar_ticks[-1], zorder=3)
    
    # Plotting the maximum point
    if peak_flag == True:
        plt.scatter(freq_vec[max_x], dir_vec_full[max_y], s=40, color='white', 
                    edgecolors='black', marker='x', linewidths=1.2, zorder=30) 
        
    # Plotting the parameters box
    if param_flag == True:
        plt.figtext(0.71, 0.78,                         
                    'Hs = ' + str(np.round(param_dict['Hs'],2)) + 'm' + '\n'
                    'Fp = ' + str(np.round(param_dict['Fp'],2)) + 'Hz' + '\n'
                    'Tp = ' + str(np.round(param_dict['Tp'],2)) + 's' + '\n'
                    'Lp = ' + str(np.round(param_dict['Lp'],2)) + 'm' + '\n'
                    'Dp = ' + str(np.round(param_dict['Dp'],2)) + '°'
                    ,fontsize=8, color='black', bbox=dict(facecolor='w', alpha=1)) 
        
    # Cutoff plot
    cutoff_res_freq = np.concatenate([[cutoff_res_freq[-1]], cutoff_res_freq])
    plt.plot(cutoff_res_freq, dir_vec_full, linestyle='--', color='red', 
             alpha=0.9, linewidth=1, zorder=30)
    plt.text(cutoff_freq + 0.002, track_angle, str(int(cutoff)) + 'm', 
             fontsize=10, color='red', fontweight='bold', zorder=30)   
    
    # Setting the ticks and ticklabels
    ax.set_xticks(freq_ticks)  
    ax.set_yticks(dir_ticks)  
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=10)
    ax.set_yticklabels(dir_tick_labels, rotation=0, fontsize=10)
    
    # Setting the parameters of labels
    plt.xlim([0, 0.5])    
    plt.tick_params(length=tick_size)
    
############################# Colorbar and title ##############################    
    
    # Setting the colorbar
    cbar = fig.colorbar(cnt1, ticks=cbar_ticks, pad=0.02)
    
    # Stablish the colorbar tick labels to be equal to the ticks
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
    
    # Adjust the colorbar ticks size    
    cbar.ax.tick_params(labelsize=10)
        
    if vec_flag == 'freq':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} Hz^{-1} \hspace{0.2} degree^{-1}})$'
    elif vec_flag == 'per':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} s \hspace{0.2} degree^{-1}})$'
    
    if norm_flag == True:
        title_unity = '(normalized)'

######################### Title and labels #################################### 

    # Adding grid
    ax.grid(which='major', color='black', alpha=0.2, linestyle='dotted')

    # Fig labels        
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel('Direction', fontsize=14, fontweight='bold')
    
    # Showing and saving the plot
    my_title = fig_title + ' ' + title_unity
    plt.title(my_title, fontsize=10, fontweight='bold', y=1.0)

    # Showing and saving the plot
    plt.show()
    plt.savefig(fname, pad_inches=0.1)
    plt.close()    
    
    
    
################################ E(kx,ky) #####################################


########################## Drawing circles function ###########################
    
def draw_circle(ax, wl, dk, align_v, align_h, add_label, 
                label_angle, label_text, fontsize, alpha,
                linestyle, color):

    """
    Draws a circle at the origin in the (kx, ky) domain corresponding to a given wavelength.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to draw the circle.
    wl : float
        The wavelength (m) corresponding to the circle to be drawn.
    dk : float
        Delta k, used for adjusting the label offset from the circle.
    align_v : str
        Vertical alignment of the label ('top', 'bottom', 'center', etc.).
    align_h : str
        Horizontal alignment of the label ('left', 'right', 'center', etc.).
    add_label : bool
        Whether to add a label next to the circle.
    label_angle : float
        Angle in radians at which to place the label relative to the circle.
    label_text : str or None
        Text label to display next to the circle (typically the wavelength).
    fontsize : int
        Font size for the label.
    alpha : float
        Transparency of the circle (0=transparent, 1=opaque).
    linestyle : str
        Line style for the circle (e.g., '--' or '-').
    color : str
        Color of the circle and label.

    Returns:
    -------
    None
        Draws the circle and (optionally) the label on the provided axis.
    """
    
    
    k_radius = 2 * np.pi / wl
    circle = plt.Circle((0, 0), k_radius, fill=False, color=color,
                        alpha=alpha, linestyle=linestyle)
    ax.add_artist(circle)
    
    if add_label:
        # angle_rad = np.radians(label_angle)
        label_offset = 0.2*dk
        x = (label_offset + k_radius) * np.sin(label_angle)
        y = (label_offset + k_radius) * np.cos(label_angle)
        text = f'{label_text}m' if label_text else ' '
        
        ax.text(x, y, text, fontsize=fontsize, color=color, alpha=0.5,
                fontweight='bold', verticalalignment=align_v, 
                horizontalalignment=align_h)

########################### Plot cutoff lines #################################

def draw_cutoff_lines(ax, K_cut, cut_off, axis_lim, my_linewidth, 
                      my_fontsize, my_color):

    """
    Draws horizontal cut-off wavenumber lines and labels at ±K_cut.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to draw the lines.
    K_cut : float
        The wavenumber cutoff value (2π / cut_off).
    cut_off : float
        The cut-off wavelength in meters.
    axis_lim : float
        The maximum absolute value of the axis limits.

    Returns:
    -------
    None
        Draws horizontal lines at ±K_cut and adds labels at those lines.
    """    

    # Plot horizontal lines at ±K_cut
    ax.hlines(K_cut, -axis_lim, axis_lim, linestyles='--', color=my_color,
              alpha=0.9, linewidth=my_linewidth, zorder=30)
    ax.hlines(-K_cut, -axis_lim, axis_lim, linestyles='--', color=my_color,
              alpha=0.9, linewidth=my_linewidth, zorder=30)
    
    # Add text labels for the cut-off wavelength
    ax.text(-axis_lim + 0.02, K_cut - 0.006, f'{int(cut_off)}m', 
             fontsize=my_fontsize, color=my_color, fontweight='bold', zorder=30)
    ax.text(-axis_lim + 0.02, -K_cut + 0.0025, f'{int(cut_off)}m',
             fontsize=my_fontsize, color=my_color, fontweight='bold', zorder=30)

########################### Wave spectrum plot ################################    

def wave_spec_kxky_plot(my_spec, k_vector, my_cmap, my_ticks, axis_tick_delta, 
                        axis_lim, theta1, theta2, cut_off_1, cut_off_2,
                        fig_xlabel, fig_ylabel, cbar_title, cbar_title_size, fig_title, 
                        axis_tick_size, axis_label_size, colorbar_tick_size, 
                        title_size, file_out, norm_flag=False, peak_flag=True):    

    """
    Plot a 2D wave spectrum in the (kx, ky) domain.
    
    This function generates a filled contour plot of a wave spectrum, showing
    energy distribution in wavenumber space. It includes options for:
    - Highlighting the peak energy
    - Drawing circles corresponding to selected wavelengths
    - Adding cutoff lines for specific wavenumber thresholds
    
    Parameters:
    ----------
    my_spec : 2D array
        The spectrum data in (kx, ky) domain.
    k_vector : 1D array
        The wavenumber grid (same for both x and y axes).
    maximum_energy : float
        The maximum energy value of the spectrum (used for normalization and colorbar).
    my_cmap : matplotlib colormap
        The colormap used for the filled contour plot.
    my_ticks : list of str
        The tick labels for both x and y axes.
    axis_tick_delta : float
        Spacing between major ticks on both axes.
    axis_lim : float
        The maximum limit for both x and y axes.
    theta1 : float
        Angle (in degrees) for the first circle label orientation.
    theta2 : float
        Angle (in degrees) for the second circle label orientation.
    peak_flag : str
        'ON' to mark the maximum energy point with a marker, otherwise ignored.
    cut_off : float
        Cut-off wavelength (m) to draw cutoff lines at ±K_cut.
    fig_xlabel : str
        Label for the x-axis.
    fig_ylabel : str
        Label for the y-axis.
    clb_wave_title : str
        Title for the colorbar (usually the energy unit).
    fig_title : str
        The main figure title.
    axis_tick_size : int
        Font size for axis tick labels.
    axis_label_size : int
        Font size for axis labels.
    colorbar_tick_size : int
        Font size for colorbar tick labels.
    title_size : int
        Font size for the main figure title.
    file_out : str
        Output filename (including extension) to save the figure.
    
    Returns:
    -------
    None
        Displays and saves the plot to the specified file.
    """

############################### Plot Parameters ############################### 
    
    # Grid
    grid_alpha = 0.2
    
    # Setting the x and y ticks and ticklabels for all plots
    major_ticks = np.arange(-axis_lim, axis_lim + axis_tick_delta/2, axis_tick_delta)
        
    # Direction of the text
    theta_text1 = math.radians(theta1)
    theta_text2 = math.radians(theta2)
    
    # Delta k for the grid
    delta_k = k_vector[1] - k_vector[0]
    
    # wavenumber cut-off
    K_cut_1 = (2*np.pi)/cut_off_1
    K_cut_2 = (2*np.pi)/cut_off_2
    
################################### levels #################################### 
    
    # Normalizing the spectrum
    if norm_flag == True:
        my_spec = my_spec/np.max(my_spec)    
           
    # Set the maximum value for ticks and levels
    if np.max(my_spec) < 10:
        my_max = np.round(np.max(my_spec),1)
    else:
        my_max = int(np.round(np.max(my_spec)))    
        
    # Set my_max = 1 if there is no energy
    if my_max < 1.0:
        my_max = 1
    
    # Define step based on my_max
    step_lookup = {1: 0.05, 2: 0.1, 5: 0.2, 10: 0.5, 20: 1, 30: 2, 
                   50: 5, 100: 10, 200: 20, 500: 50, 1000: 100, 
                   2000: 200, 10000: 1000}
    step = next((v for k, v in step_lookup.items() if my_max <= k), 2000)
    
    # Creating the ticks and tick labels
    cbar_ticks = np.arange(step,my_max + step, step)
    cbar_ticks = np.round(cbar_ticks, 1) if step < 1 else cbar_ticks
    cbar_label = []    
        
    for i in range(len(cbar_ticks)):
        if (i % 2) == 0:
            cbar_label.append(str(cbar_ticks[i]))
        else:
            cbar_label.append(' ') 
        
    # The step for the contourf plot    
    cntrf_step = step/2  
    first_level = cntrf_step 
        
#################################### Plot #####################################  
    
    # figure 1
    fig, ax = plt.subplots()
    
    # Creating the levels
    contourf_levels = np.arange(cntrf_step, cbar_ticks[-1]+cntrf_step, cntrf_step)
    contourf_levels = np.round(contourf_levels,2)
    line_levels = np.arange(step, my_max+step, step)   
     
    # Plotting the lines
    plt.contour(k_vector, k_vector, my_spec, colors='black', 
                levels=[first_level], alpha=0.7, linewidths=0.2, zorder=20)
    
    plt.contour(k_vector, k_vector, my_spec, colors='black', 
                levels=line_levels, alpha=0.7, linewidths=0.2, zorder=20)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(k_vector, k_vector, my_spec, cmap=my_cmap, 
                       levels=contourf_levels, alpha=0.7, zorder=15)    
    
    # Plotting the maximum point
    if peak_flag == True:
        [max_y, max_x] = np.where(my_spec == np.max(my_spec))
        plt.scatter(k_vector[max_x[0]], k_vector[max_y[0]], s=40, 
                    color='white', marker='x', linewidths=1.2, zorder=30) 
    
    # Set the ticks and ticklabels
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticklabels(my_ticks, rotation=0, fontsize=axis_tick_size)
    ax.set_yticklabels(my_ticks, rotation=0, fontsize=axis_tick_size)
    
    # Set the axes limits
    plt.xlim([-axis_lim, axis_lim])
    plt.ylim([-axis_lim, axis_lim])
    
    # Set the ticks length 
    plt.tick_params(length=0)

    # Setting the colorbar
    cbar = fig.colorbar(cnt1, ticks=cbar_ticks, pad=0.02)
    
    # Stablish the colorbar tick labels to be equal to the ticks
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
    
    # Adjust the colorbar ticks size    
    cbar.ax.tick_params(labelsize=10)
    
    # Colorbar title
    cbar.ax.set_title(cbar_title, horizontalalignment='center', fontsize=cbar_title_size, 
                     fontweight='bold', x=0.7, y=0.998)

    # Adding grid
    ax.grid(which='major', color='black', alpha=grid_alpha, linestyle='dotted')

    # Drawing the circles
    draw_circle(ax, wl=70, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=False, label_angle=theta_text1, label_text=None, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=100, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=100, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=150, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=150, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
        
    draw_circle(ax, wl=200, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=200, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=300, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=300, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=600, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=600, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=1000, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=1000, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    # Cutoff lines plot
    if cut_off_1:
        draw_cutoff_lines(ax, K_cut_1, cut_off_1, axis_lim, my_linewidth=1, 
                          my_fontsize=12, my_color='red')
    # if cut_off_2:
    #     draw_cutoff_lines(ax, K_cut_2, cut_off_2, axis_lim, my_linewidth=1, 
    #                       my_fontsize=12, my_color='blue')


    # Fig labels        
    plt.xlabel(fig_xlabel, fontsize=axis_label_size, fontweight='bold')
    plt.ylabel(fig_ylabel, fontsize=axis_label_size, fontweight='bold')
    plt.title(fig_title, fontsize=title_size, fontweight='bold')

    # Showing and saving the plot
    plt.show()
    plt.savefig(file_out)
    plt.close('all')     
    
############## Real part of a cross image spectrum plot #######################    

def real_cross_spec_kxky_plot(my_spec, k_vector, my_cmap, my_ticks, 
                              axis_tick_delta, axis_lim, theta1, theta2, 
                              cut_off_1, cut_off_2, fig_xlabel, fig_ylabel, cbar_title, 
                              cbar_title_size, fig_title, axis_tick_size, axis_label_size, 
                              colorbar_tick_size, title_size, file_out, 
                              norm_flag=False, peak_flag=True):   

    """
    Plot the real part of a cross image spectrum in the (kx, ky) domain.
    
    This function generates a filled contour plot of the real part of a 
    cross image spectrum, showing energy distribution in wavenumber space. 
    It includes options for:
    - Highlighting the peak energy
    - Drawing circles corresponding to selected wavelengths
    - Adding cutoff lines for specific wavenumber thresholds
    
    Parameters:
    ----------
    my_spec : 2D array
        The spectrum data in (kx, ky) domain.
    k_vector : 1D array
        The wavenumber grid (same for both x and y axes).
    maximum_energy : float
        The maximum energy value of the spectrum (used for normalization and colorbar).
    my_cmap : matplotlib colormap
        The colormap used for the filled contour plot.
    my_ticks : list of str
        The tick labels for both x and y axes.
    axis_tick_delta : float
        Spacing between major ticks on both axes.
    axis_lim : float
        The maximum limit for both x and y axes.
    theta1 : float
        Angle (in degrees) for the first circle label orientation.
    theta2 : float
        Angle (in degrees) for the second circle label orientation.
    peak_flag : str
        'ON' to mark the maximum energy point with a marker, otherwise ignored.
    cut_off : float
        Cut-off wavelength (m) to draw cutoff lines at ±K_cut.
    fig_xlabel : str
        Label for the x-axis.
    fig_ylabel : str
        Label for the y-axis.
    clb_wave_title : str
        Title for the colorbar (usually the energy unit).
    fig_title : str
        The main figure title.
    axis_tick_size : int
        Font size for axis tick labels.
    axis_label_size : int
        Font size for axis labels.
    colorbar_tick_size : int
        Font size for colorbar tick labels.
    title_size : int
        Font size for the main figure title.
    file_out : str
        Output filename (including extension) to save the figure.
    
    Returns:
    -------
    None
        Displays and saves the plot to the specified file.
    """

############################### Plot Parameters ############################### 
    
    # The spectrum
    real_spec = np.real(my_spec)
    imag_spec = np.imag(my_spec)

    # Grid
    grid_alpha = 0.2
    
    # Setting the x and y ticks and ticklabels for all plots
    major_ticks = np.arange(-axis_lim, axis_lim + axis_tick_delta/2, axis_tick_delta)
        
    # Direction of the text
    theta_text1 = math.radians(theta1)
    theta_text2 = math.radians(theta2)
    
    # Delta k for the grid
    delta_k = k_vector[1] - k_vector[0]
    
    # wavenumber cut-off
    K_cut_1 = (2*np.pi)/cut_off_1
    K_cut_2 = (2*np.pi)/cut_off_2
    
############## Peak value without 180° directional ambiguity ################## 

    # Set a small tolerance to catch small numerical differences
    tol = 1e-5
    
    # Step 1: Find all real peaks (within tolerance)
    real_max_energy = np.max(real_spec)
    real_max_coords = np.column_stack(np.where(np.abs(real_spec - real_max_energy) <= tol))

    # Step 2: Evaluate 3x3 neighborhood means in the imaginary part
    def get_window_mean(matrix, y, x):
        y_min = max(y - 1, 0)
        y_max = min(y + 2, matrix.shape[0])
        x_min = max(x - 1, 0)
        x_max = min(x + 2, matrix.shape[1])
        window = matrix[y_min:y_max, x_min:x_max]
        return np.mean(window)
    
    # Step 3: Loop and select the best peak
    best_coord = None
    best_mean = -np.inf  # We want the most positive region
    
    for coord in real_max_coords:
        y, x = coord
        mean_val = get_window_mean(imag_spec, y, x)
        if mean_val > best_mean and mean_val > 0:
            best_mean = mean_val
            best_coord = coord
    
    # Step 4: Fallback if no positive mean region is found
    if best_coord is None:
        print("Warning: No real peak corresponds to a positive region in the imaginary part.")
        best_coord = real_max_coords[0]  # fallback to the first max
    
    # Final selected peak coordinates
    max_y, max_x = best_coord
        
###################### Creating ticks and tick labels #########################    
    
# ############################################################################### 

#     # Maximum value and coordinates of the imaginary part of the cross spectrum    
#     imag_maximum_energy = np.max(imag_spec)
#     imag_maximum_coords = np.column_stack(np.where(imag_spec == imag_maximum_energy))
#     imag_maximum_coord = imag_maximum_coords[0]
    
#     # Set a small tolerance to catch small numerical differences
#     tol = 1e-5
    
#     # Find all coordinates with values *within the tolerance* of the maximum
#     real_maximum_energy = np.max(real_spec)
#     real_maximum_coords = np.column_stack(np.where(np.abs(real_spec - real_maximum_energy) <= tol))
       
#     # Find the real maximum coordinate closest to the imaginary maximum coordinate
#     distances = np.sqrt((real_maximum_coords[:, 0] - imag_maximum_coord[0])**2 +
#                         (real_maximum_coords[:, 1] - imag_maximum_coord[1])**2)
    
#     closest_index = np.argmin(distances)
#     closest_real_maximum_coord = real_maximum_coords[closest_index]  
        
#     # Maximum coordinates
#     max_x = closest_real_maximum_coord[0]
#     max_y = closest_real_maximum_coord[1]
    
################################### levels #################################### 
    
    # Normalizing the spectrum
    if norm_flag == True:
        real_spec = real_spec/np.max(real_spec)    
           
    # Set the maximum value for ticks and levels
    if np.max(real_spec) < 10:
        my_max = np.round(np.max(real_spec),1)
    else:
        my_max = int(np.round(np.max(real_spec)))    
        
    # Set my_max = 1 if there is no energy
    if my_max < 1.0:
        my_max = 1
    
    # Define step based on my_max
    step_lookup = {1: 0.05, 2: 0.1, 5: 0.2, 10: 0.5, 20: 1, 30: 2, 
                   50: 5, 100: 10, 200: 20, 500: 50, 1000: 100, 
                   2000: 200, 10000: 1000}
    step = next((v for k, v in step_lookup.items() if my_max <= k), 2000)
    
    # Creating the ticks and tick labels
    cbar_ticks = np.arange(step,my_max + step, step)
    cbar_ticks = np.round(cbar_ticks, 1) if step < 1 else cbar_ticks
    cbar_label = []    
        
    for i in range(len(cbar_ticks)):
        if (i % 2) == 0:
            cbar_label.append(str(cbar_ticks[i]))
        else:
            cbar_label.append(' ') 
        
    # The step for the contourf plot    
    cntrf_step = step/2  
    first_level = cntrf_step 
    
#################################### Plot #####################################  
    
    # figure 1
    fig, ax = plt.subplots()   
       
    # Creating the levels
    contourf_levels = np.arange(cntrf_step, cbar_ticks[-1]+cntrf_step, cntrf_step)
    contourf_levels = np.round(contourf_levels,2)
    line_levels = np.arange(step, my_max+step, step)       
   
    # Plotting the lines
    plt.contour(k_vector, k_vector, real_spec, colors='black', 
                levels=[first_level], alpha=0.7, linewidths=0.2, zorder=4)
    
    plt.contour(k_vector, k_vector, real_spec, colors='black', 
                levels=line_levels, alpha=0.7, linewidths=0.2, zorder=4)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(k_vector, k_vector, real_spec, cmap=my_cmap, 
                       levels=contourf_levels, alpha=0.7, zorder=3)    
    
    # Plotting the maximum point
    if peak_flag == True:
        # [max_y, max_x] = np.where(real_spec == np.max(real_spec))
        plt.scatter(k_vector[max_x], k_vector[max_y], s=40, 
                    color='white', marker='x', linewidths=1.2, zorder=30) 
    
    # Set the ticks and ticklabels
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticklabels(my_ticks, rotation=0, fontsize=axis_tick_size)
    ax.set_yticklabels(my_ticks, rotation=0, fontsize=axis_tick_size)
    
    # Set the axes limits
    plt.xlim([-axis_lim, axis_lim])
    plt.ylim([-axis_lim, axis_lim])
    
    # Set the ticks length 
    plt.tick_params(length=0)

    # Setting the colorbar
    cbar = fig.colorbar(cnt1, ticks=cbar_ticks, pad=0.02)
    
    # Stablish the colorbar tick labels to be equal to the ticks
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
    
    # Adjust the colorbar ticks size    
    cbar.ax.tick_params(labelsize=10)
    
    # Colorbar title
    cbar.ax.set_title(cbar_title, horizontalalignment='center', fontsize=cbar_title_size, 
                     fontweight='bold', x=0.7, y=0.998)

    # Adding grid
    ax.grid(which='major', color='black', alpha=grid_alpha, linestyle='dotted')

    # Drawing the circles
    draw_circle(ax, wl=70, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=False, label_angle=theta_text1, label_text=None, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=100, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=100, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=150, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=150, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
        
    draw_circle(ax, wl=200, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=200, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=300, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=300, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=600, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=600, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=1000, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=1000, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    # Cutoff lines plot
    if cut_off_1:
        draw_cutoff_lines(ax, K_cut_1, cut_off_1, axis_lim, my_linewidth=1, 
                          my_fontsize=12, my_color='red')
    # if cut_off_2:
    #     draw_cutoff_lines(ax, K_cut_2, cut_off_2, axis_lim, my_linewidth=1, 
    #                       my_fontsize=12, my_color='blue')

    # Fig labels        
    plt.xlabel(fig_xlabel, fontsize=axis_label_size, fontweight='bold')
    plt.ylabel(fig_ylabel, fontsize=axis_label_size, fontweight='bold')
    plt.title(fig_title, fontsize=title_size, fontweight='bold')

    # Showing and saving the plot
    plt.show()
    plt.savefig(file_out)
    plt.close('all')     

############ Imaginary part of a cross image spectrum plot ####################    

def imag_cross_spec_kxky_plot(my_spec, k_vector, my_cmap, my_ticks, 
                              axis_tick_delta, axis_lim, theta1, theta2, 
                              cut_off_1, cut_off_2, fig_xlabel, fig_ylabel, cbar_title, 
                              cbar_title_size, fig_title, axis_tick_size, axis_label_size, 
                              colorbar_tick_size, title_size, file_out, 
                              norm_flag=False, peak_flag=True):   

    """
    Plot the imaginary part of a cross image spectrum in the (kx, ky) domain.
    
    This function generates a filled contour plot of the imaginary part of a 
    cross image spectrum, showing energy distribution in wavenumber space. 
    It includes options for:
    - Highlighting the peak energy
    - Drawing circles corresponding to selected wavelengths
    - Adding cutoff lines for specific wavenumber thresholds
    
    Parameters:
    ----------
    my_spec : 2D array
        The spectrum data in (kx, ky) domain.
    k_vector : 1D array
        The wavenumber grid (same for both x and y axes).
    maximum_energy : float
        The maximum energy value of the spectrum (used for normalization and colorbar).
    my_cmap : matplotlib colormap
        The colormap used for the filled contour plot.
    my_ticks : list of str
        The tick labels for both x and y axes.
    axis_tick_delta : float
        Spacing between major ticks on both axes.
    axis_lim : float
        The maximum limit for both x and y axes.
    theta1 : float
        Angle (in degrees) for the first circle label orientation.
    theta2 : float
        Angle (in degrees) for the second circle label orientation.
    peak_flag : str
        'ON' to mark the maximum energy point with a marker, otherwise ignored.
    cut_off : float
        Cut-off wavelength (m) to draw cutoff lines at ±K_cut.
    fig_xlabel : str
        Label for the x-axis.
    fig_ylabel : str
        Label for the y-axis.
    clb_wave_title : str
        Title for the colorbar (usually the energy unit).
    fig_title : str
        The main figure title.
    axis_tick_size : int
        Font size for axis tick labels.
    axis_label_size : int
        Font size for axis labels.
    colorbar_tick_size : int
        Font size for colorbar tick labels.
    title_size : int
        Font size for the main figure title.
    file_out : str
        Output filename (including extension) to save the figure.
    
    Returns:
    -------
    None
        Displays and saves the plot to the specified file.
    """

############################### Plot Parameters ############################### 
        
    # The spectrum
    real_spec = np.real(my_spec)
    imag_spec = np.imag(my_spec)
        
    # Grid
    grid_alpha = 0.2
    
    # Setting the x and y ticks and ticklabels for all plots
    major_ticks = np.arange(-axis_lim, axis_lim + axis_tick_delta/2, axis_tick_delta)
        
    # Direction of the text
    theta_text1 = math.radians(theta1)
    theta_text2 = math.radians(theta2)
    
    # Delta k for the grid
    delta_k = k_vector[1] - k_vector[0]
    
    # wavenumber cut-off
    K_cut_1 = (2*np.pi)/cut_off_1
    K_cut_2 = (2*np.pi)/cut_off_2
        
############## Peak value without 180° directional ambiguity ################## 

    # Set a small tolerance to catch small numerical differences
    tol = 1e-5
    
    # Step 1: Find all real peaks (within tolerance)
    real_max_energy = np.max(real_spec)
    real_max_coords = np.column_stack(np.where(np.abs(real_spec - real_max_energy) <= tol))

    # Step 2: Evaluate 3x3 neighborhood means in the imaginary part
    def get_window_mean(matrix, y, x):
        y_min = max(y - 1, 0)
        y_max = min(y + 2, matrix.shape[0])
        x_min = max(x - 1, 0)
        x_max = min(x + 2, matrix.shape[1])
        window = matrix[y_min:y_max, x_min:x_max]
        return np.mean(window)
    
    # Step 3: Loop and select the best peak
    best_coord = None
    best_mean = -np.inf  # We want the most positive region
    
    for coord in real_max_coords:
        y, x = coord
        mean_val = get_window_mean(imag_spec, y, x)
        if mean_val > best_mean and mean_val > 0:
            best_mean = mean_val
            best_coord = coord
    
    # Step 4: Fallback if no positive mean region is found
    if best_coord is None:
        print("Warning: No real peak corresponds to a positive region in the imaginary part.")
        best_coord = real_max_coords[0]  # fallback to the first max
    
    # Final selected peak coordinates
    max_y, max_x = best_coord
          
################################### levels #################################### 
    
    # Normalizing the spectrum
    if norm_flag == True:
        imag_spec = imag_spec/np.max(imag_spec)    
           
    # Set the maximum value for ticks and levels
    if np.max(imag_spec) < 10:
        my_max = np.ceil(np.max(imag_spec))
    else:
        # my_max = int(np.round(np.max(imag_spec)))    
        my_max = int(np.ceil(np.max(imag_spec)))
        
    # Set my_max = 1 if there is no energy
    if my_max < 1.0:
        my_max = 1
        
    # Set the minimum energy value
    my_min = -my_max
    
    #Lookup table for tick step
    step_lookup = {1: 0.2, 2: 0.4, 5: 1, 10: 2, 20: 4, 30: 5, 50: 10, 100: 20, 
                   200: 40, 500: 100, 1000: 200, 2000: 400, 10000: 2000}
    
    # Set tick_step, cntrf_step and cntr_step
    step = next((v for k, v in step_lookup.items() if my_max <= k), 4000)
    
    # Creating the ticks and tick labels
    cbar_ticks_pos = np.arange(step, my_max + step/2, step)
    cbar_ticks_neg = -1*cbar_ticks_pos[::-1]
    cbar_ticks = np.concatenate((cbar_ticks_neg, [0], cbar_ticks_pos))
    cbar_ticks = np.round(cbar_ticks, 1) if step < 1 else cbar_ticks
    cbar_label = []   
        
    for i in range(len(cbar_ticks)):
        if (i % 2) == 0:
            cbar_label.append(str(cbar_ticks[i]))
        else:
            cbar_label.append(' ') 

    # Set the minimum energy value
    my_min = -my_max
                
#################################### Plot #####################################  
    
    # figure 1
    fig, ax = plt.subplots()
    
    # Creating steps
    contourf_step = step/4  
    cntr_step = step/2     
    
    # Create contourf and contour levels
    contourf_levels_pos = np.arange(contourf_step, my_max + contourf_step/2, contourf_step)
    contourf_levels_neg = -1*contourf_levels_pos[::-1]
    contourf_levels = np.concatenate((contourf_levels_neg, [0], contourf_levels_pos))
    
    cntr_levels_pos = np.arange(cntr_step, my_max + cntr_step/2, cntr_step)
    cntr_levels_neg = -1*cntr_levels_pos[::-1]
    cntr_levels = np.concatenate((cntr_levels_neg, [0], cntr_levels_pos))    
    
    # Plotting the lines
    plt.contour(k_vector, k_vector, imag_spec, colors='black', 
                levels=[cntr_step], alpha=0.7, linewidths=0.2, 
                vmin=my_min, vmax=my_max, zorder=4)
     
    plt.contour(k_vector, k_vector, imag_spec, colors='black', 
                levels=cntr_levels, alpha=0.7, linewidths=0.2, 
                vmin=my_min, vmax=my_max, zorder=4)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(k_vector, k_vector, imag_spec, cmap=my_cmap, 
                        levels=contourf_levels, alpha=0.7, vmin=cbar_ticks[0], 
                        vmax=cbar_ticks[-1], zorder=3)    
    
    # Plotting the maximum point
    if peak_flag == True:
        # [max_y, max_x] = np.where(imag_spec == np.max(imag_spec))
        plt.scatter(k_vector[max_x], k_vector[max_y], s=40, 
                    color='white', marker='x', linewidths=1.2, zorder=30)    
           
    # Set the ticks and ticklabels
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticklabels(my_ticks, rotation=0, fontsize=axis_tick_size)
    ax.set_yticklabels(my_ticks, rotation=0, fontsize=axis_tick_size)
    
    # Set the axes limits
    plt.xlim([-axis_lim, axis_lim])
    plt.ylim([-axis_lim, axis_lim])
    
    # Set the ticks length 
    plt.tick_params(length=0)

    # Setting the colorbar
    cbar = fig.colorbar(cnt1, ticks=cbar_ticks, pad=0.02)
    
    # Stablish the colorbar tick labels to be equal to the ticks
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
    
    # Adjust the colorbar ticks size    
    cbar.ax.tick_params(labelsize=10)
    
    # Colorbar title
    cbar.ax.set_title(cbar_title, horizontalalignment='center', fontsize=cbar_title_size, 
                     fontweight='bold', x=0.7, y=0.998)

    # Adding grid
    ax.grid(which='major', color='black', alpha=grid_alpha, linestyle='dotted')

    # Drawing the circles
    draw_circle(ax, wl=70, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=False, label_angle=theta_text1, label_text=None, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=100, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=100, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=150, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=150, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
        
    draw_circle(ax, wl=200, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=200, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=300, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=300, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=600, dk=delta_k, align_v='top', align_h='center', 
                add_label=True, label_angle=theta_text2, label_text=600, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    draw_circle(ax, wl=1000, dk=delta_k, align_v='bottom', align_h='center', 
                add_label=True, label_angle=theta_text1, label_text=1000, 
                fontsize=6, alpha=0.2, linestyle='--', color='black')
    
    # Cutoff lines plot
    if cut_off_1:
        draw_cutoff_lines(ax, K_cut_1, cut_off_1, axis_lim, my_linewidth=1, 
                          my_fontsize=12, my_color='red')

    # if cut_off_2:
    #     draw_cutoff_lines(ax, K_cut_2, cut_off_2, axis_lim, my_linewidth=1, 
    #                       my_fontsize=12, my_color='blue')        

    # Fig labels        
    plt.xlabel(fig_xlabel, fontsize=axis_label_size, fontweight='bold')
    plt.ylabel(fig_ylabel, fontsize=axis_label_size, fontweight='bold')
    plt.title(fig_title, fontsize=title_size, fontweight='bold')

    # Showing and saving the plot
    plt.show()
    plt.savefig(file_out)
    plt.close('all')     


