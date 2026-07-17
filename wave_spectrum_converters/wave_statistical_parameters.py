#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:33:10 2024

Script/Library: wave_statistical_parameters.py

Author: Yuri Brasil

e-mail: yuri.brasil@oceanica.ufrj.br

Modification: May 2, 2025

Objective: Calculate statistical parameters to compare any data (modeled) with 
           reference data (in situ or satellite data)

           
References: 
    
[Bias]

.Bryant et. al. (2016) - Evaluation Statistics Computed for the Wave 
Information Studies (WIS) (pp.2)    

[Normalized Bias]
    
.Ardhuin et. al. (2010) - Semiempirical Dissipation Source Functions for Ocean Waves. Part I: Definition,
Calibration, and Validation (pp. 1930)    

[Mean Absolute Error (MAE)]

[Normalized Mean Absolute Error (NMAE)]

[Root Mean Square Error (RMSE)] 

.Bryant et. al. (2016) - Evaluation Statistics Computed for the Wave 
Information Studies (WIS) (pp.2)  
.Wang and Wang (2022) - Evaluation of the ERA5 Significant Wave Height
against NDBC Buoy Data from 1979 to 2019 (pp. 156)
    
[Normalized Root Mean Square Error (RMSE)]  
    
.Bryant et. al. (2016) - Evaluation Statistics Computed for the Wave 
Information Studies (WIS) (pp.2)  
.Ardhuin et. al. (2010) - Semiempirical Dissipation Source Functions for Ocean 
Waves. Part I: Definition, Calibration, and Validation (pp. 1930) 

[Scatter Index (SI)]

.Bryant et. al. (2016) - Evaluation Statistics Computed for the Wave 
Information Studies (WIS) (pp.2)  

[Mentaschi Scatter Index (MSI)]
    
.Mentaschi et. al. (2013) - Problems in RMSE-based wave model validations    
.Campos et. al. (2020) - Extreme Wind and Wave Predictability From Operational 
Forecasts at the Drake Passage

[FOEX]

[r]

[R²]

[Tendency line]

"""

###############################################################################

import numpy as np
from scipy import stats

# Define a function to calculate the Bias between reference data and input data.
def bias(reference_data, data):
    
    # Check if reference_data is a NumPy array and store the result in ref_flag.
    ref_flag = isinstance(reference_data, np.ndarray)

    # Check if data is a NumPy array and store the result in data_flag.
    data_flag = isinstance(data, np.ndarray)

    # Ensure reference_data and data have the same number of elements.
    if len(reference_data) != len(data):
        raise ValueError("Reference data and data must have the same number of elements.")    
    
    # Determine the number of elements in reference_data.
    n_data = len(reference_data)
    
    # If reference_data is not a NumPy array, convert it to one.
    if not ref_flag:
        reference_data = np.array(reference_data)

    # If data is not a NumPy array, convert it to one.
    if not data_flag:
        data = np.array(data)
        
    # Compute the difference between the data and the reference data element-wise.
    diff = data - reference_data     
        
    # Calculate the bias (mean difference) by summing the differences 
    # and dividing by the number of elements.
    bias = np.sum(diff) / n_data
        
    # Return the calculated bias.
    return bias

###############################################################################

# Define a function to calculate the Normalized Bias between reference data and input data.
def norm_bias(reference_data,data):
    
    # Check if reference_data is a NumPy array and store the result in ref_flag.
    ref_flag = isinstance(reference_data, np.ndarray)

    # Check if data is a NumPy array and store the result in data_flag.
    data_flag = isinstance(data, np.ndarray)

    # Ensure reference_data and data have the same number of elements.
    if len(reference_data) != len(data):
        raise ValueError("Reference data and data must have the same number of elements.")    
    
    # Determine the number of elements in reference_data.
    n_data = len(reference_data)
    
    # If reference_data is not a NumPy array, convert it to one.
    if not ref_flag:
        reference_data = np.array(reference_data)

    # If data is not a NumPy array, convert it to one.
    if not data_flag:
        data = np.array(data)
        
    # Compute the difference between the data and the reference data element-wise.
    diff = data - reference_data   

    # Calculate the normalized bias (normalized mean difference) by summing the differences 
    # and dividing by the sum of reference elements.            
    norm_bias = np.sum(diff)/np.sum(reference_data)
        
    return norm_bias

###############################################################################

# Define a function to calculate the Mean Absolute Error (MAE) between reference data and input data.
def mae(reference_data, data):
    
    # Check if reference_data is a NumPy array and store the result in ref_flag.
    ref_flag = isinstance(reference_data, np.ndarray)

    # Check if data is a NumPy array and store the result in data_flag.
    data_flag = isinstance(data, np.ndarray)

    # Ensure reference_data and data have the same number of elements.
    if len(reference_data) != len(data):
        raise ValueError("Reference data and data must have the same number of elements.")    
    
    # Determine the number of elements in reference_data.
    n_data = len(reference_data)
    
    # If reference_data is not a NumPy array, convert it to one.
    if not ref_flag:
        reference_data = np.array(reference_data)

    # If data is not a NumPy array, convert it to one.
    if not data_flag:
        data = np.array(data)
        
    # Compute the absolute difference between the data and the reference data element-wise.
    diff = abs(data - reference_data)     
        
    # Calculate the MAE (mean absolute difference) by summing the differences and 
    # dividing by the number of elements.
    mae = np.sum(diff) / n_data
        
    # Return the calculated MAE.
    return mae

###############################################################################

# Define a function to calculate the Normalized Mean Absolute Error (NMAE) 
# between reference data and input data.
def norm_mae(reference_data, data):
    
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
        
    # Compute the absolute difference between the data and the reference data element-wise.
    diff = abs(data - reference_data)     
        
    # Calculate the MAE (mean absolute difference) by summing the differences and 
    # dividing by the sum of the squared reference data. 
    norm_mae = np.sum(diff) / np.sum(reference_data**2)
        
    # Return the calculated Normalized MAE.
    return norm_mae

###############################################################################

# Define a function to calculate the Root Mean Square Error (RMSE) 
# between reference data and input data.
def rmse(reference_data,data):

    # Check if reference_data is a NumPy array and store the result in ref_flag.
    ref_flag = isinstance(reference_data, np.ndarray)

    # Check if data is a NumPy array and store the result in data_flag.
    data_flag = isinstance(data, np.ndarray)

    # Ensure reference_data and data have the same number of elements.
    if len(reference_data) != len(data):
        raise ValueError("Reference data and data must have the same number of elements.")    
    
    # Determine the number of elements in reference_data.
    n_data = len(reference_data)
    
    # If reference_data is not a NumPy array, convert it to one.
    if not ref_flag:
        reference_data = np.array(reference_data)

    # If data is not a NumPy array, convert it to one.
    if not data_flag:
        data = np.array(data)        
        
    # Absolute difference between analysed data and reference data:
    diff = data - reference_data     
        
    # Calculate the RMSE:
    # 1. Square the differences (diff**2).
    # 2. Compute the sum of the squared differences (np.sum(diff**2)).
    # 3. Divide by the number of elements (n_data) to get the mean of the squared differences.
    # 4. Take the square root of the mean to calculate RMSE.
    rmse = np.sqrt(np.sum(diff**2) / n_data)
    
    # Return the calculated RMSE.
    return rmse

###############################################################################

# Define a function to calculate the normalized Root Mean Square Error (nRMSE) between reference data and input data.
def norm_rmse(reference_data, data):

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
        
    # Compute the difference between data and reference data element-wise.
    diff = data - reference_data    

    # Calculate the normalized RMSE:
    # 1. Square the differences (diff**2).
    # 2. Compute the sum of the squared differences (np.sum(diff**2)).
    # 3. Compute the sum of the squared reference data (np.sum(reference_data**2)).
    # 4. Divide the sum of squared differences by the sum of squared reference data.
    # 5. Take the square root to calculate the normalized RMSE.
    norm_rmse = np.sqrt(np.sum(diff**2) / np.sum(reference_data**2))
    
    # Return the calculated normalized RMSE.
    return norm_rmse

###############################################################################

# Define a function to calculate the Scatter Index (SI) between reference data and input data.
def scatter_index(reference_data, data):
    
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
        
    # Calculate the Root Mean Square Error (RMSE) between reference_data and data
    # using the previously defined `rmse` function.
    calculated_rmse = rmse(reference_data, data)
    
    # Calculate the Scatter Index (SI):
    # Divide the RMSE by the mean of the input data.
    scatter_index = calculated_rmse / np.mean(data)
    
    # Return the calculated Scatter Index.
    return scatter_index

###############################################################################

# Define a function to calculate the "Ment" Scatter Index between reference data and input data.
def ment_scatter_index(reference_data, data):
     
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
        
    # Compute the difference between data and reference data with respect to their means:
    # 1. Subtract the mean of `data` from each element of `data`.
    # 2. Subtract the mean of `reference_data` from each element of `reference_data`.
    # 3. Compute the difference between the adjusted `data` and `reference_data`.
    diff = (data - np.mean(data)) - (reference_data - np.mean(reference_data))    
    
    # Calculate the "Ment" Scatter Index:
    # Divide the sum of the computed differences by the sum of the squared reference data.
    ment_scatter_index = np.sqrt(np.sum(diff**2) / np.sum(reference_data**2))
    
    # Return the calculated "Ment" Scatter Index.
    return ment_scatter_index 
    
    
###############################################################################

# Define a function to calculate the Factor of exceedance (FOEX) between reference data and input data.
def foex(reference_data, data):
     
    # Check if reference_data is a NumPy array and store the result in ref_flag.
    ref_flag = isinstance(reference_data, np.ndarray)

    # Check if data is a NumPy array and store the result in data_flag.
    data_flag = isinstance(data, np.ndarray)
    
    # Ensure reference_data and data have the same number of elements.
    if len(reference_data) != len(data):
        raise ValueError("Reference data and data must have the same number of elements.")  

    # Determine the number of elements in reference_data.
    n_data = len(reference_data)

    # If reference_data is not a NumPy array, convert it to one.
    if not ref_flag:
        reference_data = np.array(reference_data)
        
    # If data is not a NumPy array, convert it to one.
    if not data_flag:
        data = np.array(data)

    # Count the number of elements in data greater than reference_data.
    exceedance_count = np.sum(data > reference_data)
    
    # Calculate the factor of exceedence
    foex = 100*(exceedance_count/n_data - 0.5)
    
    return foex
    
###############################################################################

# Calculate the Pearson coeficient and other linear regression parameters
def r_coefficient(reference_data, data):
     
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

    # Linear regression values
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(reference_data, data)
    
    return slope, intercept, rvalue, pvalue, stderr     
    
    
def r2_coefficient(reference_data, data):    
    
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

    # Linear regression values
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(reference_data, data)
    
    # Tendency line
    tl = intercept + slope * reference_data 

    # Squared root error
    sq_error = np.sum((data-tl)**2)
    
    # Variance
    sq_var = np.sum((data-np.mean(data))**2) 
    
    # R² 
    r_squared = 1 - sq_error/sq_var
    
    return r_squared  
        
def tendency_line(reference_data, data, x_vector):    
    
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

    # Linear regression values
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(reference_data, data)
    
    # Difference between last and first elements
    x_range = x_vector[-1] - x_vector[0]

    # Find the order of magnitude
    order = np.floor(np.log10(x_range))
    
    # Define step as a fraction of that order
    if x_range > 1:
        step = 10**(order - 1)
    else:
        step = 10**(order - 2)     
        
    # High resolution tendency line x values 
    tl_x = np.arange(x_vector[0], x_vector[-1]+step, step) 
            
    # High resolution tendency line y values
    tl_y = intercept + slope * tl_x 
    
    return tl_x, tl_y    
    