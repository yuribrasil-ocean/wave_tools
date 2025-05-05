#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon May  5 16:10:25 2025

# Script: sar_image_spectrum_plots.py

# Updated by: Yuri Brasil

# e-mail: yuri.brasil@oceanica.ufrj.br

# Modification: May 5, 2025

# Objective: Plot different forms of cross density spectrum (f,theta)

Functions:
    
    real_cartesian_spec - Function to plot the 2D contourf graph of the real
                          part of a cross image spectrum.
                     
    map_spec - Function to plot the 2D contourf graph of the wave spectrum
               as a heat map, showing the values on the graph.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

########################## Setting Figure parameters #########################

# Figure size and quality
mpl.rcParams['figure.figsize'] = (7, 5)
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.dpi'] = 300

# Don't cut nothing in the figure saved
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'

################## Cross image spectrum plot functions ########################

############ Real part of cross image spectrum cartesian plot #################  

def real_cross_cartesian_spec(dir_vec, freq_vec, spec2d, param_dict, fname, 
                              my_cmap, norm_flag, peak_flag, param_flag, 
                              vec_flag, fig_title, tick_size): 

############################ Adjusting parameters #############################
    
    mpl.rcParams['figure.figsize'] = (8, 5)
    
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
    
    # Define step based on my_max
    step_lookup = {1: 0.05, 2: 0.1, 5: 0.2, 10: 0.5, 20: 1, 30: 2, 50: 5}
    step = next((v for k, v in step_lookup.items() if my_max <= k), 10)
    
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
    first_level = cntrf_step #0.5*step       

################################## The Plot ###################################

    # figure 1
    fig, ax = plt.subplots()
       
    # Creating the levels
    cmax_levels = np.arange(cntrf_step, cbar_ticks[-1]+cntrf_step, cntrf_step)
    cmax_levels = np.round(cmax_levels,2)
    my_levels = np.arange(step, my_max+step, step)   
    
    # Plotting the lines
    plt.contour(freq_vec, dir_vec_full, spec2d_full, colors='black', 
                levels=[first_level], alpha=0.8, linewidths=0.2, zorder=20)
    
    plt.contour(freq_vec, dir_vec_full, spec2d_full, colors='black', 
                levels=my_levels, alpha=0.8, linewidths=0.2, zorder=20)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(freq_vec, dir_vec_full, spec2d_full, cmap=my_cmap, 
                       levels=cmax_levels, alpha=0.95, zorder=15)
    
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
        
    # Setting the ticks and ticklabels
    ax.set_xticks(freq_ticks)
    # ax.set_yticks(dir_vec_full)    
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

########## imaginary part of cross image spectrum cartesian plot ##############  

def imag_cross_cartesian_spec(dir_vec, freq_vec, spec2d, param_dict, fname, 
                              my_cmap, norm_flag, peak_flag, param_flag, 
                              vec_flag, fig_title, tick_size): 

############################ Adjusting parameters #############################
    
    mpl.rcParams['figure.figsize'] = (8, 5)
    
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
        
    # Maximum energy value
    my_max_im = np.round(np.max(spec2d_full),4)
    if (my_max_im >=30) and (my_max_im % 10 != 0):
        my_max_im = int(np.round(np.max(spec2d_full),-1))
           
    # Define step based on my_max
    step_lookup = {1: 0.1, 2: 0.2, 5: 0.5, 10: 1, 20: 2, 30: 3, 50: 5}
    im_step = next((v for k, v in step_lookup.items() if my_max_im <= k), 10)
    
    my_max_im = my_max_im + im_step
    my_min_im = - my_max_im
    
    # Creating the ticks and tick labels
    cbar_ticks = np.arange(-my_max_im,my_max_im + im_step, im_step)
    cbar_ticks = np.round(cbar_ticks, 1) if im_step < 1 else cbar_ticks
    cbar_label = []  
        
    for i in range(len(cbar_ticks)):
        if (i % 2) == 0:
            cbar_label.append(str(cbar_ticks[i]))
        else:
            cbar_label.append(' ') 
        
    # The step for the contourf plot    
    cntrf_step = im_step/2  
    first_level = cntrf_step #0.5*step       

################################## The Plot ###################################

    # figure 1
    fig, ax = plt.subplots()
       
    # Creating the levels
    cmax_levels = np.arange(cbar_ticks[0], cbar_ticks[-1]+cntrf_step, cntrf_step)
    cmax_levels = np.round(cmax_levels,2)
    my_levels = np.arange(my_min_im, my_max_im + im_step, im_step)  
    
    # Plotting the lines
    plt.contour(freq_vec, dir_vec_full, spec2d_full, colors='black', 
                levels=[first_level], alpha=0.8, linewidths=0.2, zorder=20)
    
    plt.contour(freq_vec, dir_vec_full, spec2d_full, colors='black', 
                levels=my_levels, alpha=0.8, linewidths=0.2, zorder=20)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(freq_vec, dir_vec_full, spec2d_full, cmap=my_cmap, 
                       levels=cmax_levels, alpha=0.95, zorder=15)
    
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
        
    # Setting the ticks and ticklabels
    ax.set_xticks(freq_ticks)
    # ax.set_yticks(dir_vec_full)    
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
    if my_max_im == cbar_ticks[-1]:
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
    