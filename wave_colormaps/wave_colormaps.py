#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:05:32 2024

# Script: wave_colormaps.py

# Author: Yuri Brasil

# e-mail: yuri.brasil@oceanica.ufrj.br

# Modification: May2, 2025

# Objective: Package with different colormaps for wave data plot purposes

"""

import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Color codes, in RGB
wind_field_rgb = np.array([(165,206,255), (74,157,255), (0,93,255),
                           (127,159,166), (190,192,121), (253,224,76),
                           (255,117,0), (225,0,0), (147,0,0), (116,0,0), 
                           (116,0,83)])/255

hs_field_rgb = np.array([(0, 55, 149),(29, 82, 169),(58, 108, 188),
                         (87, 135, 208),(116, 161, 227),(151, 181, 205),
                         (186, 201, 182),(220, 220, 160),(255, 240, 137),
                         (252, 215, 104),(250, 190, 72),(247, 165, 39),
                         (244, 140, 6),(238, 117, 5),(232, 94, 4),
                         (226, 70, 3),(220, 47, 2),(202, 35, 2),
                         (184, 24, 2),(165, 12, 2),(147, 0, 2)])/255


real_part_spec_rgb = np.array([(235,255,255), (165,206,255), (74,157,255), 
                               (0,93,255), (127,159,166), (190,192,121), 
                               (253,224,76), (255,117,0), (225,0,0), 
                               (147,0,0)])/255

wave_spec_rgb = np.array([(255,255,255), (51,163,255), (36,103,255), 
                          (0,41,143), (69,0,99), (55,6,23), (157,2,8), 
                          (220,47,2), (244,140,6), 
                          (255,186,8)])/255


# imag_part_spec_rgb = np.array([(0,0,200), (0,54,233), (28,111,255),
#                                (70,160,255), (160,209,255),
#                                (195,0,0), (255,28,0), (223,54,54),
#                                (228,89,89), (231,107,107)])/255


# Edit the "seismic color map
base_cmap = cm.get_cmap('seismic', 256)

# Define cut points
lower = base_cmap(np.linspace(0.0, 0.45, 128))   # first 45%
upper = base_cmap(np.linspace(0.55, 1.0, 128))   # last 45%

# Concatenate (skip the white center)
new_colors = np.vstack((lower, upper))

# Creating colormaps
wind_field_cmap = LinearSegmentedColormap.from_list('wind_field', 
                                                  wind_field_rgb, N=81)

hs_field_cmap = LinearSegmentedColormap.from_list('hs_field', 
                                                  hs_field_rgb, N=121)

real_part_spec_cmap = LinearSegmentedColormap.from_list('sar_spec', 
                                                  real_part_spec_rgb, N=81)

# Create new colormap
imag_part_spec_cmap = mcolors.ListedColormap(new_colors, name='seismic_no_white')
# imag_part_spec_cmap = LinearSegmentedColormap.from_list('imag_spec', 
#                                                   imag_part_spec_rgb, N=81)

wave_spec_cmap = LinearSegmentedColormap.from_list('wave_spec', 
                                                  wave_spec_rgb, N=81)













