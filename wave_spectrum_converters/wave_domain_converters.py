#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:05:39 2023

Script/Function: wave_spectral_converters.py

Author: Yuri Brasil

e-mail: yuri.brasil@oceanica.ufrj.br

Modification: June 22, 2025

Objective: 
           
References: 

    
"""

###############################################################################

import numpy as np
from scipy.interpolate import RegularGridInterpolator

################## Calculating the d vector (df or dk) ########################

def d_logarithmic(spec_vector):
    
    # Creating an empty array and the logarithmic ratio
    d_vec = np.zeros(len(spec_vector))
    
    # Loop to calculate the df elements (except the first and the last ones)
    for i in range(1,len(spec_vector)-1):
        d_vec[i] = (spec_vector[i+1]-spec_vector[i-1])/2

    # Calculating the first and last elements  
    d_vec[0] = spec_vector[1] - spec_vector[0]
    d_vec[-1] = spec_vector[-1] - spec_vector[-2]
    
    return d_vec

################## Calculating the d vector (df or dk) ########################

def d_logarithmic_old(spec_vector,increment):
    
    # Creating an empty array and the logarithmic ratio
    d_vec = np.zeros(len(spec_vector))
    log_ratio = 1 + increment
    
    # Loop to calculate the df elements (except the first and the last ones)
    for i in range(1,len(spec_vector)-1):
        d_vec[i] = (spec_vector[i+1]-spec_vector[i-1])/2

    # Calculating the first and last elements
    d_vec[0] = (spec_vector[1] - spec_vector[0]/log_ratio)/2
    d_vec[-1] = (spec_vector[-1]*log_ratio - spec_vector[-2])/2

    return d_vec

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

################# Converting E(f,θ) to E(k,θ) an vice-versa ###################

def wave_spec_conversion_f_k(dir_vec, f_or_k_vec, spec, conversion_flag):
    
    # Gravity acceleration
    g = 9.80665
    
    if conversion_flag == 'to_f':
        
        # Wavenumber, wavelength and frequency vectors
        k_vec = f_or_k_vec
        L_vec = (2*np.pi)/k_vec
        f_vec = np.sqrt(g/(L_vec*2*np.pi))
        
        # Check the orientation        
        spec2d = spec_orientation(dir_vec, k_vec, spec)
        
        # Conversion jacobian (8pi²f)/g
        jacobian = ((8*np.pi**2)*f_vec)/g
        
        # Adjust spectrum
        converted_spec = spec2d * jacobian[np.newaxis,:]
        
        # Frequency vector to be returned
        f_or_k_vec = f_vec
        
    elif conversion_flag == 'to_k':
        
        # Frequency, wavelength and wavenumber vectors
        f_vec = f_or_k_vec
        L_vec = g/(2*np.pi*f_vec**2)
        k_vec = 2*np.pi/L_vec
        
        # Check the orientation        
        spec2d = spec_orientation(dir_vec, f_vec, spec)
        
        # Conversion jacobian (1/4pi)*sqr(g/k)
        jacobian = (1/(4*np.pi))*np.sqrt(g/k_vec)
        
        # Adjust spectrum
        converted_spec = spec2d * jacobian[np.newaxis,:]

        # Frequency vector to be returned
        f_or_k_vec = k_vec    
    
    return converted_spec, f_or_k_vec, dir_vec            

#################### Interpolation of wave spectrum grid ######################

def wave_spec_interpolation(dir_vec, freq_vec, wave_spectrum,
                            dir_vec_new, freq_vec_new,
                            log_freq_vec=True, 
                            interp_method='linear'):    
    
    # Check orientation
    wave_spectrum = spec_orientation(dir_vec, freq_vec, wave_spectrum)
    
    # Handle direction wrapping (0° to 360°)
    dir_vec = np.array(dir_vec) % 360
    dir_vec_new = np.array(dir_vec_new) % 360
    
    # Loop convert into log interpolation in log space if requested
    if log_freq_vec:
        freq_coord_orig = np.log(freq_vec)
        freq_coord_new = np.log(freq_vec_new)
    else:
        freq_coord_orig = freq_vec
        freq_coord_new = freq_vec_new
    
    # Create the interpolator
    interpolator = RegularGridInterpolator((dir_vec, freq_coord_orig),
                                           wave_spectrum, method=interp_method,
                                           bounds_error=False, fill_value=0.0)

   
    # Create new coordinate grids
    dir_grid, freq_grid = np.meshgrid(dir_vec_new, freq_coord_new, indexing='ij')
    
    # Interpolate
    interpolated_spectrum = interpolator((dir_grid, freq_grid))
    
    
    # Difference vectors
    dtheta = np.ones(len(dir_vec))*(dir_vec[1]-dir_vec[0])
    dtheta_new = np.ones(len(dir_vec_new))*(dir_vec_new[1]-dir_vec_new[0]) 
    df = d_logarithmic(freq_vec)
    df_new = d_logarithmic(freq_vec_new)
    
    # Interpolated energy
    interp_energy = np.sum(interpolated_spectrum * np.outer(dtheta_new, df_new))
    
    # Original energy
    original_energy = np.sum(wave_spectrum * np.outer(dtheta, df))
    
    # Adjust the energy
    interpolated_spectrum *= original_energy / interp_energy
   
    
    return interpolated_spectrum


############### Interpolation of semilog wave spectrum grid ###################

def wave_spec_semilog_interpolation(y_in, x_in, dy_in, dx_in, matrix, y_out, 
                                    x_out,  dy_out, dx_out, energy_flag=True):
    
    matrix = spec_orientation(y_in, x_in, matrix)

    # Step 1: Wrap direction (rows) for 0°/360° continuity
    dy_deg = np.rad2deg(dy_in[0])
    matrix_ext = np.vstack([matrix[-1:, :], matrix, matrix[:1, :]])
    y_in_ext = np.concatenate([[y_in[0] - dy_deg], y_in, [y_in[-1] + dy_deg]])

    # Step 2: Prepare log-linear axes
    x_inb = x_in
    x_in_linb = np.log(x_inb)
    x_out_lin = np.log(x_out)

    # Step 3: Apply Jacobian (log → lin)
    jacobian_log2lin = x_inb
    matrix_log2lin = matrix_ext * jacobian_log2lin[np.newaxis, :]

    # Step 4: Interpolation using RegularGridInterpolator
    interpolator = RegularGridInterpolator(
        (y_in_ext, x_in_linb),
        matrix_log2lin,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # Create output mesh and interpolate
    Y_out_grid, X_out_grid = np.meshgrid(y_out, x_out_lin, indexing='ij')
    points_out = np.column_stack((Y_out_grid.ravel(), X_out_grid.ravel()))
    matrix_interp = interpolator(points_out).reshape(len(y_out), len(x_out))

    # Step 5: Apply inverse Jacobian (lin → log)
    j_lin2log = np.exp(-x_out_lin)
    new_spectrum = matrix_interp * j_lin2log[np.newaxis, :]

    # Step 6: Energy adjustment
    if energy_flag:
        energy_interp = np.sum(new_spectrum * np.outer(dy_out, dx_out))
        original_energy = np.sum(matrix * np.outer(dy_in, dx_in))
        if energy_interp > 0:
            new_spectrum *= original_energy / energy_interp

    return new_spectrum