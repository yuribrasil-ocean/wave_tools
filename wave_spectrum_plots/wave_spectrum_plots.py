#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Saturday September 5th 2020

# Script: wave_spectrum_plots.py

# Author: Felipe Santos - felipems@gmail.com

# Updated by: Yuri Brasil

# e-mail: yuri.brasil@oceanica.ufrj.br

# Modification: May 2, 2025

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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

########################## Setting Figure parameters #########################

# Figure size and quality
mpl.rcParams['figure.figsize'] = (7, 5)
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.dpi'] = 300

# Don't cut nothing in the figure saved
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'

########################## Spectrum plot functions ############################

################## Omnidirectional Wave Spectrum Plot #########################

def omnidirectional_spec(dir_vec, freq_vec, my_spec, param_dict, fname, my_cmap, 
                         norm_flag, peak_flag, param_flag, vec_flag, fig_title):

############################ Adjusting parameters #############################

    # Verifying the number of dimensions of the input wave spectrum    
    if np.array(my_spec).ndim == 2:
        d_theta = np.deg2rad(dir_vec[1]-dir_vec[0])
        my_spec = np.sum(my_spec*d_theta,axis=0)        
    
    # Normalizing the spectrum
    if norm_flag == True:
        my_spec = my_spec/np.max(my_spec)    
        
    # Setting up the maximum value for y ticks
    if np.max(my_spec) < 10:
        my_max = np.round(np.max(my_spec),1)
    else:
        my_max = int(np.round(np.max(my_spec)))

    # Define step based on my_max
    step_lookup = {1: 0.1, 2: 0.2, 5: 0.5, 10: 1, 20: 2, 50: 5}
    step = next((v for k, v in step_lookup.items() if my_max <= k), 20)

    # Creating the ticks and tick labels
    y_ticks = np.arange(0, my_max + step, step)
    y_ticks = np.round(y_ticks, 1) if step < 1 else y_ticks 
    
    # Creating the unity string    
    if vec_flag == 'freq':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} Hz^{-1}})$'
    elif vec_flag == 'per':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} s})$'
    
    # Creating the 'normalized' string
    if norm_flag == True:
        title_unity = '(normalized)'
    
################################## The Plot ###################################

    # figure 1
    fig, ax = plt.subplots()           
    
    # Plotting the lines
    plt.plot(freq_vec, my_spec, color='royalblue',
                linewidth=0.8, zorder=20)
    
    # Filling the plot with color
    plt.fill_between(freq_vec, my_spec, color='royalblue',
                     alpha=0.6, zorder=15)
    
    # Plotting the peak vertical line
    if peak_flag == True:
        max_x = np.where(my_spec == np.max(my_spec))
        plt.vlines(freq_vec[max_x], 0, my_max + step, color='black',
                   linewidth=0.8, linestyle='dashed', zorder=25)
        
    # Plotting the parameters box
    if param_flag == True:
        plt.figtext(0.83, 0.78,                         
                    'Hs = ' + str(np.round(param_dict['Hs'],2)) + 'm' + '\n'
                    'Fp = ' + str(np.round(param_dict['Fp'],2)) + 'Hz' + '\n'
                    'Tp = ' + str(np.round(param_dict['Tp'],2)) + 's' + '\n'
                    'Lp = ' + str(np.round(param_dict['Lp'],2)) + 'm' + '\n'
                    'Dp = ' + str(np.round(param_dict['Dp'],2)) + '°'
                    ,fontsize=8, color='black', bbox=dict(facecolor='w', alpha=1))     

###################### Creating ticks and tick labels #########################
    
    # Preparing the vec_ticks (frequency and period)
    freq_ticks = np.arange(0.0,0.5,0.05)
    
    freq_tick_labels = []
    for i in freq_ticks:
        freq_tick_labels.append(str(np.round(i,2))+'Hz')
        
    period_tick_labels = []
    for i in freq_ticks:
        period_tick_labels.append(str(np.round(1/i,1))+'s')

######################## Adjusting ticks and labels ###########################

    # Creating the ticks of period vector
    if vec_flag == 'per':
        x_tick_labels = period_tick_labels
        x_label = 'Period'
    elif vec_flag == 'freq':
        x_tick_labels = freq_tick_labels
        x_label = 'Frequency'

    # Setting the ticks and ticklabels
    ax.set_xticks(freq_ticks)
    ax.set_yticks(y_ticks)        
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=9)
    ax.set_yticklabels(y_ticks, rotation=0, fontsize=9)
    
    # Setting the parameters of labels
    plt.xlim([freq_vec[0], freq_vec[-1]]) 
    plt.ylim([0, my_max+step])
    plt.tick_params(length=0)
    
    # Adding grid
    ax.grid(which='major', color='black', alpha=0.2, linestyle='dotted')

    # Fig labels        
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    my_title = fig_title + ' ' + title_unity
    plt.title(my_title, fontsize=10, fontweight='bold', y=1.0)
    
    # Showing and saving the plot
    plt.show()
    plt.savefig(fname, pad_inches=0.1)
    plt.close()

######################### Polar Wave Spectrum Plot ############################

def polar_spec(dir_vec, freq_vec, spec2d, param_dict, fname, ftick, my_cmap,
               arrow_flag, radius_flag, norm_flag, peak_flag, param_flag, 
               vec_flag, fig_title, fhalf=False):        
    
############################ Adjusting parameters #############################

    # Editing the spectrum and direction vector to fill the trigonometric 
    # circle and transforming the direction elements into radians
    spec2d_full = np.vstack((spec2d,spec2d[0,:]))
    dir_rad_vec = np.deg2rad(dir_vec)
    dir_rad_vec_full = np.hstack((dir_rad_vec,np.deg2rad(360))) 

    # set the grid radius, theta
    [r, th] = np.meshgrid(freq_vec, dir_rad_vec_full)
    
    if norm_flag == True:
        spec2d_full = spec2d_full/np.max(spec2d_full)

    # Creating the ticks of period vector
    if vec_flag == 'per':
        ptick = np.ceil(1/np.array(ftick))
        
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
    first_level = cntrf_step     

################################## The Plot ###################################

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')

    # adjusting to North/wind rose
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")    
        
    # Creating the levels
    cmax_levels = np.arange(cntrf_step, cbar_ticks[-1]+cntrf_step, cntrf_step)
    cmax_levels = np.round(cmax_levels,2)
    my_levels = np.arange(step, my_max+step, step)   
    
    # Plotting the lines
    plt.contour(th, r, spec2d_full, colors='k', levels=[first_level], alpha=0.8,
                linewidths = 0.2, zorder=20)
    
    plt.contour(th, r, spec2d_full, colors='k', levels=my_levels, alpha=0.8,
                linewidths = 0.2, zorder=20)

    # Plotting the spectrum contourf
    im1 = ax.contourf(th, r, spec2d_full, cmap=my_cmap,
                      levels=cmax_levels, alpha=0.95, zorder=15)
    
    # Plotting the maximum point
    if peak_flag == True:
        [max_y, max_x] = np.where(spec2d_full == np.max(spec2d_full))
        plt.scatter(dir_rad_vec_full[max_y[0]], freq_vec[max_x[0]], s=40, 
                    color='white', marker='2', linewidths=1.2, zorder=30) 

    # theta grid
    ax.set_thetagrids(angles=np.arange(0, 360, 45),
                      labels=[' ', '45°', '90°', '135°','180°','225°','270°','315°'],
                      fontweight='bold')
    
######################### Editing the concentric circles ######################

    # Polar parameters
    track=360
    dirlabel1=315
    dirlabel2=135
    
    if vec_flag == 'freq':
        # Radius (freq) grid
        ax.set_ylim(ymin=0.)
        if ftick is not None:
            rlabel = ['{:.02f}'.format(elm) for elm in ftick]
            rlabel[-1] += 'Hz'
            if fhalf:
                rlabel[::2] = ''
                       
            ax.set_rgrids(ftick, rlabel, alpha=0)
            ax.set_ylim(ymax=ftick[-1])
            for i in range(len(ftick)):
                elm = ftick[i]
                if elm == ftick[-1]:
                    if radius_flag == True:    
                        ax.text((dirlabel1+2)*np.pi/180, elm-0.015, "{:.2f}".format(elm) + 'Hz',
                                ha='center', va='top', fontsize=8, zorder=20)
                        ax.text((dirlabel2+2)*np.pi/180, elm-0.02, "{:.2f}".format(elm) + 'Hz',
                                ha='center', va='top', fontsize=8, zorder=20)
                else:
                    if radius_flag == True:    
                        ax.text(dirlabel1*np.pi/180, elm-0.01, "{:.2f}".format(elm) + 'Hz',
                                ha='center', va='bottom', fontsize=8, zorder=20)
                        ax.text(dirlabel2*np.pi/180, elm-0.01, "{:.2f}".format(elm) + 'Hz',
                                ha='center', va='top', fontsize=8, zorder=20)
                        
    elif vec_flag == 'per':
        # Radius (freq) grid
        ax.set_ylim(ymin=0.)
        if ftick is not None:
            rlabel = [str(int(elm)) for elm in ptick]
            rlabel[-1] += 's'
            if fhalf:
                rlabel[::2] = ''
                       
            ax.set_rgrids(ftick, rlabel, alpha=0)
            ax.set_ylim(ymax=ftick[-1])
            for i in range(len(ftick)):
                elm = ftick[i]
                if elm == ftick[-1]:
                    if radius_flag == True:    
                        ax.text((dirlabel1+2)*np.pi/180, elm-0.005, rlabel[i],
                                ha='center', va='top', fontsize=8, zorder=20)
                        ax.text(dirlabel2*np.pi/180, elm-0.01, rlabel[i],
                                ha='center', va='top', fontsize=8, zorder=20)
                else:
                    if radius_flag == True:    
                        ax.text(dirlabel1*np.pi/180, elm, rlabel[i] + 's',
                                ha='center', va='bottom', fontsize=8, zorder=20)
                        ax.text(dirlabel2*np.pi/180, elm, rlabel[i] + 's',
                                ha='center', va='top', fontsize=8, zorder=20)          
      
    # style of gridlines
    ax.tick_params(labelsize=8)
    gkws = dict(linestyle='--', color='.6', lw=.5)  # zorder does not work
    ax.grid(axis='x', **gkws)  # theta gridlines
    ls = ax.get_ygridlines()  # freq gridlines
    plt.setp(ls[1::2], **gkws)  # -
    plt.setp(ls[0::2], dashes=[2, 2], **gkws)  # --

############################# Colorbar and title ##############################

    # colorbar
    cbar = fig.colorbar(im1, ticks=cbar_ticks)
    
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
        
    cbar.ax.tick_params(labelsize=10)
    
    if vec_flag == 'freq':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} Hz^{-1} \hspace{0.2} degree^{-1}})$'
    elif vec_flag == 'per':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} s \hspace{0.2} degree^{-1}})$'
    
    if norm_flag == True:
        title_unity = '(normalized)'
        
    # Showing and saving the plot
    my_title = fig_title + ' ' + title_unity
    plt.title(my_title, fontsize=10, fontweight='bold', y=1.0)   
                
############################### Ploting the arrows ############################

    if arrow_flag == True:

        # 0º - 180º arrow
        plt.arrow(np.deg2rad(track), 0, 0, 0.188, alpha=1, width=0,
                  edgecolor='black', facecolor='black', lw=0.8,
                  head_width=0.05, head_length=0.01, zorder=10)
    
        plt.arrow(np.deg2rad(track-180), 0, 0, 0.188, alpha=1, width=0,
                  edgecolor='black', facecolor='black', lw=0.8,
                  head_width=0.05, head_length=0.01, zorder=10)    
    
        # 90º - 270º arrow
        plt.arrow(np.deg2rad(track+90), 0, 0, 0.188, alpha=1, width=0,
                  edgecolor='black', facecolor='black', lw=0.8,
                  head_width=0.05, head_length=0.01, zorder=10)
    
        plt.arrow(np.deg2rad(track-90), 0, 0, 0.188, alpha=1, width=0,
                  edgecolor='black', facecolor='black', lw=0.8,
                  head_width=0.05, head_length=0.01, zorder=10)

###############################################################################
    
    # Showing and saving the plot
    plt.show()
    plt.savefig(fname, pad_inches=0.1)
    plt.close()

##################### Cartesian Wave Spectrum Plot ############################  

def cartesian_spec(dir_vec, freq_vec, spec2d, param_dict, fname, my_cmap, 
                   norm_flag, peak_flag, param_flag, vec_flag, fig_title): 

############################ Adjusting parameters #############################
    
    mpl.rcParams['figure.figsize'] = (8, 5)
    
    # Editing the spectrum and direction vector to fill the trigonometric 
    # circle
    spec2d_full = np.vstack((spec2d,spec2d[0,:]))
    dir_vec_full = np.hstack((dir_vec,360)) 

    if norm_flag == True:
        spec2d_full = spec2d_full/np.max(spec2d_full)
        
###################### Creating ticks and tick labels #########################

    # Preparing the dir_step lookup table
    dir_step_lookup = {(25, 73): 2, (73, 200): 3, (200, 500): 10, (500, 800): 20}
    
    # Creating the dir_step
    dir_step = next((v for (start, end), v in dir_step_lookup.items() if start < len(dir_vec_full) <= end), 50)

    # Preparing the dir_tick_labels
    dir_tick_labels = [' ']*len(dir_vec_full)
    
    for d in range(dir_step,len(dir_vec_full),dir_step):
        if len(dir_vec_full) <= 361:
            dir_tick_labels[d] = str(int(dir_vec_full[d])) + '°'
        else:
            dir_tick_labels[d] = str(np.round(dir_vec_full[d],1)) + '°'    
    
    # Preparing the vec_ticks
    freq_ticks = np.arange(0.0,0.5,0.05)
    
    freq_tick_labels = []
    for i in freq_ticks:
        freq_tick_labels.append(str(np.round(i,2))+'Hz')
        
    period_tick_labels = []
    for i in freq_ticks:
        period_tick_labels.append(str(np.round(1/i,1))+'s')
    
    period_tick_labels[0] = ' '

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
    plt.contour(freq_vec, dir_vec_full, spec2d_full, colors='k', levels=[first_level], alpha=0.8,
                linewidths=0.2, zorder=20)
    
    plt.contour(freq_vec, dir_vec_full, spec2d_full, colors='k', levels=my_levels, alpha=0.8,
                linewidths=0.2, zorder=20)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(freq_vec, dir_vec_full, spec2d_full, 
                       cmap=my_cmap, levels=cmax_levels,
                       alpha=0.95, zorder=15)
    
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
    ax.set_yticks(dir_vec_full)    
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=10)
    ax.set_yticklabels(dir_tick_labels, rotation=0, fontsize=10)
    
    # Setting the parameters of labels
    plt.xlim([0, 0.5])    
    plt.tick_params(length=0)
    
############################# Colorbar and title ##############################    
    
    # Setting the colorbar
    # colorbar
    cbar = fig.colorbar(cnt1, ticks=cbar_ticks, pad=0.02)
    
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
        
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

######################## Map Wave Spectrum Plot ################################     

def map_spec(dir_vec, freq_vec, spec2d, param_dict, fname, my_cmap, 
             norm_flag, peak_flag, param_flag, vec_flag, fig_title): 

############################ Adjusting parameters #############################
    
    mpl.rcParams['figure.figsize'] = (8, 5)

    # set the grid radius, theta 
    freq_vec = freq_vec

    if norm_flag == True:
        spec2d = spec2d/np.max(spec2d)
        
###################### Creating ticks and tick labels #########################

    # Preparing the dir_ticks
    dir_ticks = dir_vec[::-1]

    # Preparing the dir_step lookup table
    dir_step_lookup = {(24, 72): 2, (72, 200): 3, (200, 500): 10, (500, 800): 20}
    
    # Creating the dir_step
    dir_step = next((v for (start, end), v in dir_step_lookup.items() if start < len(dir_ticks) <= end), 50)

    # Preparing the dir_tick_labels
    dir_tick_labels = [' ']*len(dir_ticks)
    
    for d in range(1,len(dir_ticks),dir_step):
        if len(dir_ticks) <= 360:
            dir_tick_labels[d] = str(int(dir_ticks[d])) + '°'
        else:
            dir_tick_labels[d] = str(np.round(dir_ticks[d],1)) + '°'  
    
    # dir_tick_labels[0] = ' '
    
    # Preparing the freq_tick_labels
    freq_step_lookup = {(25, 70): 5, (73, 200): 10, 
                        (200, 500): 20, (500, 1001): 80}

    freq_step = next((v for (start, end), v in freq_step_lookup.items() if start < len(freq_vec) <= end), 2)
    
    # Adjusting linewidth based on frequency step
    my_linewidth = {5: 0.7, 10: 0.5, 20: 0.2, 80: 0.1}.get(freq_step, 1)
    

###############################################################################

    # Creating the ticks of period vector
    
    freq_tick_labels = [' ']*len(freq_vec)
    period_tick_labels = [' ']*len(freq_vec)
    
    if vec_flag == 'per':
        x_tick_labels = period_tick_labels
        x_label = 'Period'
        
        for f in range(0,len(freq_vec),freq_step):
            period_tick_labels[f] = str(np.round(1/freq_vec[f],1)) + 's'
        
        
    elif vec_flag == 'freq':
        x_tick_labels = freq_tick_labels
        x_label = 'Frequency'
        
        for f in range(0,len(freq_vec),freq_step):
            freq_tick_labels[f] = str(np.round(freq_vec[f],2)) + 'Hz'
        
        
    # Setting up the maximum value for ticks and levels
    if np.max(spec2d) < 10:
        my_max = np.round(np.max(spec2d),1)
    else:
        my_max = int(np.round(np.max(spec2d)))
    
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
        
    # Inverting the spectrum    
    spec2d = spec2d[::-1,:]

    im_freq_vec = np.arange(len(freq_vec))
    im_dir_vec = np.arange(len(dir_vec))

################################## The Plot ###################################

    # figure 1
    fig, ax = plt.subplots()
    map1 = ax.imshow(spec2d, cmap=my_cmap, vmin=0, vmax=my_max,
                     interpolation='nearest', aspect='auto', zorder=10)    
    
    if  len(freq_vec) <= 30:
    # Plotting the values on the map
        for i in range(np.size(spec2d,0)):
            for j in range(np.size(spec2d,1)):
                ax.text(j, i, np.round(spec2d[i][j],1), ha='center', va='center',
                        color='white', fontsize='xx-small',fontweight='bold', zorder=20)

    # Creating the grid by hand   
    # First lines        
    plt.axvline(x=-0.50, linewidth=my_linewidth, linestyle='-', color='black', zorder=30)
    plt.axhline(y=-0.50, linewidth=my_linewidth, linestyle='-', color='black', zorder=30)
    
    # The whole grid
    for k in range(np.size(spec2d,1)):
        plt.axvline(x=k+0.5, linewidth=my_linewidth, linestyle='-', color='black', zorder=30)
    for k in range(np.size(spec2d,0)):    
        plt.axhline(y=k+0.5, linewidth=my_linewidth, linestyle='-', color='black', zorder=30)
    
    # Plotting the maximum point
    if peak_flag == True:
        [max_y, max_x] = np.where(spec2d == np.max(spec2d))
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((im_freq_vec[max_x[0]]-0.5,
                                         im_dir_vec[max_y[0]]-0.5), 1, 1,
                                        fill=None, linestyle='-', linewidth=1.4,
                                        edgecolor='white', zorder=40))
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
    ax.set_xticks(np.arange(len(freq_vec)))
    ax.set_yticks(np.arange(len(dir_vec)))
       
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=8)
    ax.set_yticklabels(dir_tick_labels, rotation=0, fontsize=10)
    
    # Setting the parameters of labels
    plt.tick_params(length=0)
    
################################ Colorbar #####################################    
    
    # Setting the colorbar
    # colorbar
    cbar = fig.colorbar(map1, ticks=cbar_ticks, pad=0.02)
    
    if my_max == cbar_ticks[-1]:
        cbar.ax.set_yticklabels(cbar_ticks)
        
    cbar.ax.tick_params(labelsize=10)
    
######################### Title and labels ####################################
        
    # Title strings    
    if vec_flag == 'freq':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} Hz^{-1} \hspace{0.2} degree^{-1}})$'
    elif vec_flag == 'per':
        title_unity = r'$(\mathbf{m^2 \hspace{0.2} s \hspace{0.2} degree^{-1}})$'
    
    if norm_flag == True:
        title_unity = '(normalized)'
     
    # Title    
    my_title = fig_title + ' ' + title_unity
    plt.title(my_title, fontsize=10, fontweight='bold', y=1.0)

    # Fig labels        
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel('Direction', fontsize=14, fontweight='bold')
        
    # Showing and saving the plot
    plt.show()
    plt.savefig(fname, pad_inches=0.1)
    plt.close()    
    