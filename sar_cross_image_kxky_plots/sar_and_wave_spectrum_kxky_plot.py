#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Script: sar_and_wave_spectrum_kxky_plot.py

# Author: Yuri Brasil - yuri.brasil@oceanica.ufrj.br

# Created on Fri May 30 19:38:06 2025

# Modification: June 2, 2025

# Objective: Plot SAR cross spectrum (real and imaginary part), exponetial term,
             and wave spectrum in (kx, ky) domain
             
# Auxiliary functions:
    
    draw_circle - function to draw circles in the figure

# Functions:
            
    plot_spec_kxky - function to plot a spectrum in (kx,ky) domain
    
    
"""

import math 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

########################## Setting Figure parameters #########################

# mpl.rcParams['figure.figsize'] = (10, 9)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['axes.labelpad'] = 12.0

# Don't cut nothing in the figure saved
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'#'pdf'
mpl.rcParams['savefig.pad_inches'] = 0.1

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

def wave_spec_kxky_plot(my_spec, k_vector, my_cmap, my_ticks,
                        axis_tick_delta, axis_lim, theta1, theta2, cut_off, 
                        fig_xlabel, fig_ylabel, clb_title, fig_title, 
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
    K_cut = (2*np.pi)/cut_off
    
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
                levels=[first_level], alpha=0.8, linewidths=0.2, zorder=20)
    
    plt.contour(k_vector, k_vector, my_spec, colors='black', 
                levels=line_levels, alpha=0.8, linewidths=0.2, zorder=20)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(k_vector, k_vector, my_spec, cmap=my_cmap, 
                       levels=contourf_levels, alpha=0.95, zorder=15)    
    
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
    if cut_off:
        draw_cutoff_lines(ax, K_cut, cut_off, axis_lim, my_linewidth=1, 
                              my_fontsize=12, my_color='red')

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
                              cut_off, fig_xlabel, fig_ylabel, clb_title, 
                              fig_title, axis_tick_size, axis_label_size, 
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
    K_cut = (2*np.pi)/cut_off
    
############################################################################### 

    # Maximum value and coordinates of the imaginary part of the cross spectrum    
    imag_maximum_energy = np.max(imag_spec)
    imag_maximum_coords = np.column_stack(np.where(imag_spec == imag_maximum_energy))
    imag_maximum_coord = imag_maximum_coords[0]
    
    # Set a small tolerance to catch small numerical differences
    tol = 1e-5
    
    # Find all coordinates with values *within the tolerance* of the maximum
    real_maximum_energy = np.max(real_spec)
    real_maximum_coords = np.column_stack(np.where(np.abs(real_spec - real_maximum_energy) <= tol))
       
    # Find the real maximum coordinate closest to the imaginary maximum coordinate
    distances = np.sqrt((real_maximum_coords[:, 0] - imag_maximum_coord[0])**2 +
                        (real_maximum_coords[:, 1] - imag_maximum_coord[1])**2)
    
    closest_index = np.argmin(distances)
    closest_real_maximum_coord = real_maximum_coords[closest_index]  
        
    # Maximum coordinates
    max_x = closest_real_maximum_coord[0]
    max_y = closest_real_maximum_coord[1]
    
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
        [max_y, max_x] = np.where(real_spec == np.max(real_spec))
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
    if cut_off:
        draw_cutoff_lines(ax, K_cut, cut_off, axis_lim, my_linewidth=1, 
                              my_fontsize=12, my_color='red')

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
                              cut_off, fig_xlabel, fig_ylabel, clb_title, 
                              fig_title, axis_tick_size, axis_label_size, 
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
    K_cut = (2*np.pi)/cut_off
    
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
                levels=[cntr_step], alpha=0.8, linewidths=0.2, 
                vmin=my_min, vmax=my_max, zorder=4)
     
    plt.contour(k_vector, k_vector, imag_spec, colors='black', 
                levels=cntr_levels, alpha=0.8, linewidths=0.2, 
                vmin=my_min, vmax=my_max, zorder=4)

    # Plotting the spectrum contourf
    cnt1 = ax.contourf(k_vector, k_vector, imag_spec, cmap=my_cmap, 
                        levels=contourf_levels, alpha=0.8, vmin=cbar_ticks[0], 
                        vmax=cbar_ticks[-1], zorder=3)    
    
    # Plotting the maximum point
    if peak_flag == True:
        [max_y, max_x] = np.where(imag_spec == np.max(imag_spec))
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
    if cut_off:
        draw_cutoff_lines(ax, K_cut, cut_off, axis_lim, my_linewidth=1, 
                              my_fontsize=12, my_color='red')

    # Fig labels        
    plt.xlabel(fig_xlabel, fontsize=axis_label_size, fontweight='bold')
    plt.ylabel(fig_ylabel, fontsize=axis_label_size, fontweight='bold')
    plt.title(fig_title, fontsize=title_size, fontweight='bold')

    # Showing and saving the plot
    plt.show()
    plt.savefig(file_out)
    plt.close('all')     

######################## 2D Exponential term plot #############################    

def efactor_2D_plot(my_efactor, k_vector, my_cmap, my_ticks,
                    axis_tick_delta, axis_lim, theta1, theta2, 
                    cut_off, fig_xlabel, fig_ylabel, 
                    fig_title, axis_tick_size, axis_label_size, 
                    colorbar_tick_size, title_size, file_out):    

    """
    Plot a 2D exponential factor in the (kx, ky) domain.
    
    This function generates a filled contour plot of a exponential factor, 
    showing its distribution in wavenumber space. It includes options for:
    - Drawing circles corresponding to selected wavelengths
    - Adding cutoff lines for specific wavenumber thresholds
    
    Parameters:
    ----------
    my_efactor : 2D array
        The exponential data in (kx, ky) domain.
    k_vector : 1D array
        The wavenumber grid (same for both x and y axes).
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
    cut_off : float
        Cut-off wavelength (m) to draw cutoff lines at ±K_cut.
    fig_xlabel : str
        Label for the x-axis.
    fig_ylabel : str
        Label for the y-axis.
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
    K_cut = (2*np.pi)/cut_off
     
################################### levels #################################### 
     
    # Creating the ticks and tick labels
    cbar_ticks = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,
                 1e-5,1e-4,1e-3,1e-2,1e-1,1e-0]
    cbar_label = []    
    
    for i in range(len(cbar_ticks)):
         cbar_label.append(str(cbar_ticks[i]))
    
    # The step for the contourf plot    
    first_level = 1e-12    
    
    # Creating the levels
    cmax_levels = cbar_ticks
    my_levels = cbar_ticks 
     
#################################### Plot #####################################  
     
    # figure 1
    fig, ax = plt.subplots()    
       
    # Plotting the first line
    plt.contour(k_vector, k_vector, my_efactor, colors='k', 
                levels=[first_level], alpha=0.8, linewidths= 0.2, vmin=0, 
                vmax=1, zorder=4)
       
    # Plotting all the other lines
    plt.contour(k_vector, k_vector, my_efactor, colors='k', levels=my_levels, 
                alpha=0.8, linewidths= 0.2, vmin=0, vmax=1, zorder=3)
       
    # Plotting the spectrum
    cnt1 = plt.contourf(k_vector, k_vector, my_efactor, cmap=my_cmap, 
                        levels=cmax_levels, vmin=first_level, vmax=1, 
                        norm=LogNorm())
        
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
    
    # Set the colorbar
    clb = plt.colorbar(cnt1, pad=0.03)
    clb.set_ticks(cbar_ticks)
    clb.ax.tick_params(labelsize=colorbar_tick_size)
    clb.ax.set_yticklabels(cbar_label)
        
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
    if cut_off:
        draw_cutoff_lines(ax, K_cut, cut_off, axis_lim, my_linewidth=1, 
                              my_fontsize=12, my_color='red')
    
    # Fig labels        
    plt.xlabel(fig_xlabel, fontsize=axis_label_size, fontweight='bold')
    plt.ylabel(fig_ylabel, fontsize=axis_label_size, fontweight='bold')
    plt.title(fig_title, fontsize=title_size, fontweight='bold')
    
    # Showing and saving the plot
    plt.show()
    plt.savefig(file_out)
    plt.close('all')     
        
