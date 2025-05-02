#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sun Apr  4 17:02:26 2021

Script/Function: wave_colormaps_example_plots.py

Author: Yuri Brasil

e-mail: yuri.brasil@oceanica.ufrj.br

Modification: May 2, 2025

"""
import numpy as np
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt
import wave_colormaps as wv_colors

# Getting the start time
start_time = datetime.now()

############################## Figure settings ################################

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['axes.labelpad'] = 6.0

# Don't cut nothing in the figure saved
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'
mpl.rcParams['savefig.pad_inches'] = 0.1

######################### Creating a matrix of directions #####################

# Matrix resolution
resolution = 128

# Median point
median_point = int(resolution/2)

# Creating an index vector that varies between -resolution/2 and resolution/2
index_vec = np.arange(1-(median_point+1),resolution+1-median_point,1)

# Creating the x and y vectors
y_vec = index_vec*0.01
x_vec = y_vec

# Replacing the median point of x and y (zero) for a small number to
# prevent further problems
y_vec[median_point] = 1E-10
x_vec[median_point] = 1E-10
    
# Creating the 2D x and y matrices and the K modulus matrix
[x_mesh, y_mesh] = np.meshgrid(x_vec,y_vec)
modulus = np.sqrt(x_mesh**2 + y_mesh**2)

# Replacing the central point by a small number
modulus[median_point,median_point] = 1E-10

# Creating the grid of directions
dir_grid = np.rad2deg(np.arctan2(x_mesh,y_mesh))
dir_grid = np.where(dir_grid < 0, dir_grid + 360.0, dir_grid)    

# Correcting the directions values, since arc tangent is not continous
dir_grid[len(dir_grid)//2,len(dir_grid)//2] = 0.0

# 0°
dir_grid[len(dir_grid)//2+1:len(dir_grid),len(dir_grid)//2] = 0.0

#90°
dir_grid[len(dir_grid)//2,len(dir_grid)//2+1:len(dir_grid)] = 90.0

#180°
dir_grid[0:len(dir_grid)//2,len(dir_grid)//2] = 180.0

#270°
dir_grid[len(dir_grid)//2,0:len(dir_grid)//2] = 270.0

######################### Plot of each wave colormap ##########################

# List of colormaps
list_of_colormaps = [wv_colors.wind_field_cmap, 
                     wv_colors.hs_field_cmap,
                     wv_colors.real_part_spec_cmap,
                     wv_colors.imag_part_spec_cmap,
                     wv_colors.wave_spec_cmap]

# List of titles
list_of_titles = ['Wind field colormap', 'Hs field colormap',
                  'Real part of SAR cross spectrum colormap',
                  'Imaginary part of SAR cross spectrum colormap',
                  'Wave spectrum colormap']

# List of filenames
list_of_filenames = ['wind_field_colormap_plot',
                     'hs_field_colormap_plot',
                     'real_part_sar_cross_spectrum_colormap_plot',
                     'imaginary_part_sar_cross_spectrum_colormap_plot',
                     'wave_spectrum_colormap_plot']

ticks = np.round(np.arange(-0.6,0.6+0.1,0.1),2)

tick_labels = []
for i in ticks:
    tick_labels.append(str(i))

tick_labels[6] = '0'

# Creating the ticks and tick labels
cbar_ticks = np.arange(0,360 + 15, 15)
    

# cbar labels
cbar_labels = []         
for c in range(len(cbar_ticks)):
    if (c % 2) == 0:
        cbar_labels.append(str(int(cbar_ticks[c]))+'°')
    else:
        cbar_labels.append(' ')

# Loop of plots
for my_colormap in list_of_colormaps:

    # Index of my_colormap in list_of_colormaps
    idx = list_of_colormaps.index(my_colormap)

    # Figure
    fig, ax = plt.subplots()   
    
    # Contourf plot
    ctnf = plt.contourf(x_vec,y_vec,dir_grid,cmap=my_colormap,levels=72)
    
    # Colorbar
    cb = fig.colorbar(ctnf, pad=0.01)
    cb.set_ticks(cbar_ticks,labels=cbar_labels)
    cb.ax.tick_params(labelsize=8, size=0)    
    
    # Ticks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([], rotation=0, fontsize=8)
    ax.set_yticklabels([], rotation=0, fontsize=8)
    ax.tick_params(length=0)
    
    # Labels and title
    plt.xlabel('X')
    plt.ylabel('Y', rotation=0)
    plt.title(list_of_titles[idx])
    
    plt.show()
    plt.savefig(list_of_filenames[idx]) 
    plt.close()


# Getting the end time and printing running time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
  
