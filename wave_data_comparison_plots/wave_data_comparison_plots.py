#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 18:16:44 2025

# Script: wave_data_comparison_plots.py

# Author: Yuri Brasil

# e-mail: yuri.brasil@oceanica.ufrj.br

# Modification: May 8, 2025

# Objective: Plot a timeseries and scatter plots of several any chosen 
             wave parameter 
             
Functions:
    
    timeseries_plot -
    
    scatter_plot -   

    density_scatter_plot -      
             
"""

import numpy as np
import matplotlib as mpl
from matplotlib import style
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

########################### Setting figure parameters #########################

# Figure size and quality
mpl.rcParams['figure.figsize'] = (7, 7)
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.dpi'] = 300

# Don't cut nothing in the figure saved
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'

style.use('ggplot')

mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.edgecolor'] = 'black' 

############################### Scatter plot ##################################

def scatter_plot(reference_data, data, data_type, my_color,  my_marker, 
                 my_marker_size, tendency_line_flag, tl_x, tl_y, my_title, 
                 my_xlabel, my_ylabel, stats_flag, stats_table, fname):
    
    # Check if reference_data is a NumPy array and store the result in ref_flag.
    ref_flag = isinstance(reference_data, np.ndarray)
    
    # Check if data is a NumPy array and store the result in data_flag.
    data_flag = isinstance(data, np.ndarray)
    
    # Ensure reference_data and data have the same number of elements.
    if len(reference_data) != len(data):
        raise ValueError("Reference data and data must have the same number of elements.")    
        
    # If reference_data is not a NumPy array, convert it to one.
    if not ref_flag:
        reference_data = np.array(reference_data)
    
    # If data is not a NumPy array, convert it to one.
    if not data_flag:
        data = np.array(data)
    
    # Setting the color backgroung
    fig, ax = plt.subplots()
    
    # Scatter plot
    plt.scatter(reference_data, data, marker=my_marker, s=30, linewidth=0.7, 
             color=my_color, edgecolor='black', alpha=0.6, zorder=3)
    
########################### Ticks and tick labels #############################

    # Customizing data vectors
    if data_type == 'wave_height':
        
        # Minimum and maximum tick values
        ticks_min = 0
        ticks_max = int(np.ceil(np.max([reference_data,data])))
        
        # Difference between ticks
        ticks_diff = ticks_max-ticks_min
        
        # Calculate the tick_step
        if ticks_diff <= 2:
            tick_step = 0.2        
        elif ticks_diff <= 4 and ticks_diff > 2:
            tick_step = 0.5
        elif ticks_diff <= 10 and ticks_diff > 4:
            tick_step = 1
            
        # Create the ticks vector    
        ticks = np.arange(0, ticks_max + tick_step, tick_step)
        
        # Create an empty list to be filled
        tick_labels = [' ']*len(ticks)
        
        # Create the tick labels step
        label_step = int(1/tick_step)

        # Filling the tick labels
        for h in range(label_step,len(ticks),label_step):
            tick_labels[h] = f'{int(ticks[h])}m' 
        
    elif data_type == 'direction':    
        
        # Minimum and maximum tick values
        ticks_min = 0
        ticks_max = 360
        
        # Tick step
        tick_step = 15
        
        # Create the ticks vector    
        ticks = np.arange(0, ticks_max + tick_step, tick_step)
        
        # Create an empty list to be filled
        tick_labels = [' ']*len(ticks)
        
        # Create the tick labels step
        label_step = 2
        
        for d in range(label_step,len(ticks),label_step):
            tick_labels[d] = f'{int(ticks[d])}°'         
        
    elif data_type == 'frequency':
        
        # Minimum and maximum tick values
        ticks_min = np.round(np.min([reference_data,data]),4)
        ticks_max = np.round(np.max([reference_data,data]),4)
        
        # Difference between ticks
        ticks_diff = ticks_max-ticks_min
        
        # Calculate the tick_step
        if ticks_diff <= 0.01:
            tick_step = 0.001        
        elif ticks_diff <= 0.05 and ticks_diff > 0.01:
            tick_step = 0.002
        elif ticks_diff <= 0.1 and ticks_diff > 0.05:
            tick_step = 0.005
        elif ticks_diff > 0.1:
            tick_step = 0.01
            
        # Create the ticks vector    
        ticks = np.arange(ticks_min, ticks_max + tick_step, tick_step)
        
        # Create an empty list to be filled
        tick_labels = [' ']*len(ticks)
        
        # Create the tick labels step
        label_step = 2

        # Filling the tick labels
        for f in range(label_step,len(ticks),label_step):
            tick_labels[f] = f'{np.round(ticks[f],2)}Hz' 
        
    elif data_type == 'period':  
        
        # Minimum and maximum tick values
        ticks_min = int(np.ceil(np.min([reference_data,data])))
        ticks_max = int(np.ceil(np.max([reference_data,data])))
        
        # Difference between ticks
        ticks_diff = ticks_max-ticks_min

        # Calculate the tick_step
        if ticks_diff <= 2:
            tick_step = 0.1        
        elif ticks_diff <= 4 and ticks_diff > 2:
            tick_step = 0.2
        elif ticks_diff <= 10 and ticks_diff > 4:
            tick_step = 0.5
        elif ticks_diff > 10:
            tick_step = 1
            
        # Create the ticks vector    
        ticks = np.arange(ticks_min, ticks_max + tick_step, tick_step)
        
        # Create an empty list to be filled
        tick_labels = [' ']*len(ticks)
        
        # Create the tick labels step
        label_step = 2

        # Filling the tick labels
        for p in range(label_step,len(ticks),label_step):
            tick_labels[p] = f'{int(ticks[p])}s'         
    

    # Plotting the identity line
    plt.plot([ticks[0], ticks[-1]], [ticks[0], ticks[-1]], 
             linestyle='--', linewidth=1.5, color='black', zorder=2)    

    # Check tendency line flag
    if tendency_line_flag == True:        
        # Plotting the tendency line
        plt.plot(tl_x, tl_y, linestyle='--', linewidth=1.5, 
                 color=my_color, zorder=2)

    # Customizing the ticks
    plt.xticks(ticks, labels=tick_labels, fontweight='bold', fontsize=8)
    plt.yticks(ticks, labels=tick_labels, fontweight='bold', fontsize=8)
    ax.tick_params(axis='both',length=0)
    
    # Setting x and y limits
    plt.xlim(xmin=ticks[0], xmax=ticks[-1])
    plt.ylim(ymin=ticks[0], ymax=ticks[-1])
    
    # Setting the X label
    plt.xlabel(my_xlabel, fontsize=12, fontweight='bold', zorder=2, color='black')
    plt.ylabel(my_ylabel, fontsize=12, fontweight='bold', zorder=2, color='black')
    
    # Setting the Title
    plt.title(my_title,fontsize=13, fontweight='bold', color='black')
    
    # Creating a box to display the statistical parameters
    if stats_flag == True:
        
        tick_size = len(ticks)
        
        if data_type == 'frequency':
            x_pos = (tick_size/6.5)/tick_size            
        else:
            x_pos = (tick_size/8)/tick_size
                
        y_pos = (6*tick_size/8)/tick_size
        
        plt.figtext(x_pos, y_pos, stats_table, fontsize=8, color='black', 
                    bbox=dict(facecolor='white', alpha=1, edgecolor='black'))        

    # Showing and saving the plot
    plt.show()
    plt.savefig(fname, pad_inches=0.1)
    plt.close()


