#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:05:39 2023

Script/Function: wave_spectral_parameters.py

Author: Yuri Brasil

e-mail: yuri.brasil@oceanica.ufrj.br

Modification: May 2, 2025

Objective: Calculate average parameters such as significant wave height 
           (Hs or SWH), mean wave period (mean frequency and mean wavelegth) 
           and mean wave direction. Also, the package calculates peak 
           parameters such as peak period (peak frequency and peak wavelegth) 
           and peak direction (based on maximum value of the 2D spectrum or 
           the maximum of a parabolic fit of an integrated 1D spectrum). 
           To calculate theses parameters, the user needs to provide the df vector
           (a vector containing the differences btween two frequency bins). If the
           vector of frequencies is logarithmic, the df_logarithmic() function
           can be used to create the df vector.
           
References: 

df:
    
Hs:
                
.SWAN Group (2022) SWAN User Manual - SWAN Cycle III version 41.45 (pp. 105)
.WW3DG (2019) User manual and system documentation of WAVEWATCH III version 6.07. The WAVEWATCH III Development Group. (pp. 110)
.ECMWF (2021) IFS Documentation CY47R3 - Part VII: ECMWF Wave model (pp.82)
.Barstow et. al. (2005) Measuring and Analysing the directional spectrum of ocean waves (pp. 19)
.Goda, Y. (2010). Random seas and design of maritime structures(pp. 40)
.Holthuijsen, L. H. Waves in oceanic and coastal waters. (pp. 70)
.Jiang et. al. (2023) Global 3-hourly wind-wave and swell data for wave climate and wave energy resource research from 1950 to 2100 (pp. 3)
.Zieger et. al. (2015) Observation-based source terms in the third-generation wave model (pp. 23)

Tm and Fm:

.SWAN Group (2022) SWAN User Manual - SWAN Cycle III version 41.45 (pp. 105)
.WW3DG (2019) User manual and system documentation of WAVEWATCH III version 6.07. The WAVEWATCH III Development Group.  (pp. 110)
.ECMWF (2021) IFS Documentation CY47R3 - Part VII: ECMWF Wave model (pp.83)
.Barstow et. al. (2005) Measuring and Analysing the directional spectrum of ocean waves (pp. 19)
.Holthuijsen, L. H. Waves in oceanic and coastal waters. (pp. 61)
.Jiang et. al. (2023) Global 3-hourly wind-wave and swell data for wave climate and wave energy resource research from 1950 to 2100 (pp. 3)
.Zieger et. al. (2015) Observation-based source terms in the third-generation wave model (pp. 23)

Dm:
            
.SWAN Group (2022) SWAN User Manual - SWAN Cycle III version 41.45 (pp. 106)
.WW3DG (2019) User manual and system documentation of WAVEWATCH III version 6.07. The WAVEWATCH III Development Group.  (pp. 110)
.ECMWF (2021) IFS Documentation CY47R3 - Part VII: ECMWF Wave model (pp.83)
.Barstow et. al. (2005) Measuring and Analysing the directional spectrum of ocean waves (pp. 19)
.Jiang et. al. (2023) Global 3-hourly wind-wave and swell data for wave climate and wave energy resource research from 1950 to 2100 (pp. 3)
.Zieger et. al. (2015) Observation-based source terms in the third-generation wave model (pp. 23)


Tp and Fp:

<Parabolic Function>

.SWAN Group (2022) SWAN User Manual - SWAN Cycle III version 41.45 (pp. 106)
.WW3DG (2019) User manual and system documentation of WAVEWATCH III version 6.07. The WAVEWATCH III Development Group.  (pp. 110)
.ECMWF (2021) IFS Documentation CY47R3 - Part VII: ECMWF Wave model (pp.83)
.Björkqvist, J. V., (2019) WAM, SWAN and WAVEWATCH III in the Finnish archipelago – the effect of spectral performance
on bulk wave parameters (pp. 59)
.Zieger et. al. (2015) Observation-based source terms in the third-generation wave model (pp. 23)

<Maximum>
.Barstow et. al. (2005) Measuring and Analysing the directional spectrum of ocean waves (pp. 9)
.Jiang et. al. (2023) Global 3-hourly wind-wave and swell data for wave climate and wave energy resource research from 1950 to 2100 (pp. 3)
                
Dp:

.SWAN Group (2022) SWAN User Manual - SWAN Cycle III version 41.45 (pp. 106)
.WW3DG (2019) User manual and system documentation of WAVEWATCH III version 6.07. The WAVEWATCH III Development Group.  (pp. 111)
.Jiang et. al. (2023) Global 3-hourly wind-wave and swell data for wave climate and wave energy resource research from 1950 to 2100 (pp. 3)

Spreading:

.SWAN Group (2022) SWAN User Manual - SWAN Cycle III version 41.45 (pp. 107)
.WW3DG (2019) User manual and system documentation of WAVEWATCH III version 6.07. The WAVEWATCH III Development Group.  (pp. 112)

    

"""

###############################################################################

import numpy as np

######################## Calculating the df vector ############################

def df_logarithmic(freq_vec,increment):
    
    # Creating an empty array and the logarithmic ratio
    df = np.zeros(len(freq_vec))
    log_ratio = 1 + increment
    
    # Loop to calculate the df elements (except the first and the last ones)
    for i in range(1,len(freq_vec)-1):
        df[i] = (freq_vec[i+1]-freq_vec[i-1])/2

    # Calculating the first and last elements
    df[0] = (freq_vec[1] - freq_vec[0]/log_ratio)/2
    df[-1] = (freq_vec[-1]*log_ratio - freq_vec[-2])/2
    
    return df

############### Correcting spectrum orientation (if needed) ###################

def spec_orientation(freq_vec, dir_vec, spec):
    
    # Checking the spectrum orientation and, if necessary, correcting 
    # the spectrum orientation that should be E(f,θ)
    if spec.shape == (len(freq_vec),len(dir_vec)):
        spec2d = spec
    elif spec.shape == (len(dir_vec),len(freq_vec)):
        spec2d = np.transpose(spec) 
    
    return spec2d

########################### AVERAGE PARAMETERS ################################

############## Calculating the Significant Wave Height (Hs) ###################

def hs_spec(freq_vec, df, dir_vec, spec, two_dim_flag):
    
             
    # Calculating the dtheta element
    dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])      
    
    # Checking the dimensions
    if two_dim_flag == True:
        
        # Checking the spectrum orientation
        spec2d = spec_orientation(freq_vec, dir_vec, spec)    
        
        # 1D wave spectrum E(f)     
        spec1d_freq = np.sum(np.multiply(spec2d,dtheta),axis=1)  
        
        # Calculating the Hs    
        hs = 4*np.sqrt(np.sum(np.multiply(spec1d_freq,df)))

    # Checking the dimensions        
    elif two_dim_flag == False:
        
        # 1d wave spectrum
        spec1d_freq = spec

        # Calculating the Hs             
        hs = 4*np.sqrt(np.sum(np.multiply(spec1d_freq,df)))
    
    return hs

######################### Calculating the mean frequency ######################

def mean_frequency(freq_vec, df, dir_vec, spec, two_dim_flag, tm_flag):

    if two_dim_flag == True:
        
        # Checking the spectrum orientation
        spec2d = spec_orientation(freq_vec, dir_vec, spec)    

        # Calculate the dtheta element
        dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])

        # 1D wave spectrum E(f)     
        spec1d_freq = np.sum(np.multiply(spec2d,dtheta),axis=1)
                
        # Calculating the spectral moments        
        m_minus_one = np.sum((freq_vec**(-1))*np.multiply(spec1d_freq,df))
        m_zero = np.sum(np.multiply(spec1d_freq,df))
        m_first = np.sum(freq_vec*np.multiply(spec1d_freq,df))
        m_second = np.sum((freq_vec**2)*np.multiply(spec1d_freq,df))    
                      
        
        # Getting the average parameters fm, tm, and lm depending on 
        # the tm formulation (tm-1, tm1 or tm2)
        if tm_flag == -1:
            tm = m_minus_one/m_zero
            fm = 1/tm
            lm = (tm**2)*1.56
            
        elif tm_flag == 1:
            tm = m_zero/m_first
            fm = 1/tm
            lm = (tm**2)*1.56
            
        elif tm_flag == 2:            
            tm = np.sqrt(m_zero/m_second)
            fm = 1/tm
            lm = (tm**2)*1.56
    
        return tm, fm, lm
    
    elif two_dim_flag == False:

        # 1D wave spectrum E(f) 
        spec1d_freq = spec
        
        # Calculating the spectral moments  
        m_minus_one = np.sum((freq_vec**(-1))*np.multiply(spec1d_freq,df))
        m_zero = np.sum(np.multiply(spec1d_freq,df))
        m_first = np.sum(freq_vec*np.multiply(spec1d_freq,df))
        m_second = np.sum((freq_vec**2)*np.multiply(spec1d_freq,df))    
        
        # Getting the average parameters fm, tm, and lm depending on 
        # the tm formulation (tm-1, tm1 or tm2)
        if tm_flag == -1:
            tm = m_minus_one/m_zero
            fm = 1/tm
            lm = (tm**2)*1.56
            
        elif tm_flag == 1:
            tm = m_zero/m_first
            fm = 1/tm
            lm = (tm**2)*1.56
            
        elif tm_flag == 2:            
            tm = np.sqrt(m_zero/m_second)
            fm = 1/tm
            lm = (tm**2)*1.56
        
        return tm, fm, lm
        
######################### Calculating the mean direction ######################

def mean_direction(freq_vec, df, dir_vec, spec, two_dim_flag):

    if two_dim_flag == True:

        # Checking the spectrum orientation
        spec2d = spec_orientation(freq_vec, dir_vec, spec)    

        # Calculating the dtheta element
        dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])
        radians_vec = np.deg2rad(dir_vec)
       
        # 1D wave spectrum E(θ)         
        # spec1d_dir = np.sum(spec2d*df[:, np.newaxis], axis=0)
        spec1d_dir = np.sum(np.multiply(spec2d,df[:, np.newaxis]), axis=0)

        # Calculating the cosine and sine terms
        a_term = np.sum(np.cos(radians_vec)*np.multiply(spec1d_dir,dtheta))
        b_term = np.sum(np.sin(radians_vec)*np.multiply(spec1d_dir,dtheta))

        # Calculating the mean direction
        dm = np.rad2deg(np.arctan2(b_term,a_term))

        # Converting negative angles
        if dm < 0:
            dm = 360 + dm

        return dm

    elif two_dim_flag == False:
       
        # Calculating the dtheta element
        dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])
        radians_vec = np.deg2rad(dir_vec) 
       
        # 1D wave spectrum E(θ) 
        spec1d_dir = spec

        # Calculating the cosine and sine terms
        a_term = np.sum(np.cos(radians_vec)*np.multiply(spec1d_dir,dtheta))
        b_term = np.sum(np.sin(radians_vec)*np.multiply(spec1d_dir,dtheta))

        # Calculating the mean direction
        dm = np.rad2deg(np.arctan2(b_term,a_term))

        # Converting negative angles
        if dm < 0:
            dm = 360 + dm

        return dm

############################# PEAK PARAMETERS #################################


####### Calculating the peak frequency by fitting a parabolic function ########

def peak_frequency(freq_vec, dir_vec, spec, two_dim_flag):
    
    if two_dim_flag == True:
        
        # Checking the spectrum orientation
        spec2d = spec_orientation(freq_vec, dir_vec, spec)    
        
        # Calculate the dtheta element
        dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])

        # 1D wave spectrum E(f)     
        spec1d_freq = np.sum(np.multiply(spec2d,dtheta),axis=1)  
        
        # Getting the frequency band of maximum value of the 
        # frequency spectrum E(f)
        [fp_idx] = np.where(spec1d_freq == np.max(spec1d_freq))
        
        # Interval of the frequency vector
        interval = int(0.1*len(freq_vec))
        
        # Begininng and end of set
        beg = fp_idx[0] - interval
        end = fp_idx[0] + interval
        
        # Creating a high resolution frequency vector:
        f_step = 0.001    
        freq_high_vec = np.arange(freq_vec[beg],freq_vec[end]+f_step,f_step)        
        
        # Fitting a parabolic function around the peak that was found 
        # ([beg:end])
        coefs = np.polyfit(freq_vec[beg:end], spec1d_freq[beg:end], 3)   
        sliced_spec_function = np.polyval(coefs, freq_high_vec)
  
        # Getting the frequency of maximum value of the frequency spectrum E(f)
        [fp_fit_idx] = np.where(sliced_spec_function == np.max(sliced_spec_function))
        
        # Getting the peak parameters fp, tp, and lp
        fp = freq_high_vec[fp_fit_idx[0]]
        tp = 1/fp
        lp = (tp**2)*1.56
        
        return fp, tp, lp  
        
    elif two_dim_flag == False:
        
        # 1D wave spectrum E(f) 
        spec1d_freq = spec

        # Getting the frequency band of maximum value of the 
        # frequency spectrum E(f)
        [fp_idx] = np.where(spec1d_freq == np.max(spec1d_freq))
        
        # Interval of the frequency vector
        interval = int(0.1*len(freq_vec))
        
        # Begininng and end of set
        beg = fp_idx[0] - interval
        end = fp_idx[0] + interval
        
        # Creating a high resolution frequency vector:
        f_step = 0.001    
        freq_high_vec = np.arange(freq_vec[beg],freq_vec[end]+f_step,f_step)        
        
        # Fitting a parabolic function around the peak that was found 
        # ([beg:end])
        coefs = np.polyfit(freq_vec[beg:end], spec1d_freq[beg:end], 3)   
        sliced_spec_function = np.polyval(coefs, freq_high_vec)
  
        # Getting the frequency of maximum value of the frequency spectrum E(f)
        [fp_fit_idx] = np.where(sliced_spec_function == np.max(sliced_spec_function))
        
        # Getting the peak parameters fp, tp, and lp
        fp = freq_high_vec[fp_fit_idx[0]]
        tp = 1/fp
        lp = (tp**2)*1.56
        
        return fp, tp, lp 

######################### Calculating the peak direction ######################

def peak_direction(freq_vec, df, dir_vec, spec):
    
        # Checking the spectrum orientation
        spec2d = spec_orientation(freq_vec, dir_vec, spec)    

        # Calculate the dtheta element
        dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])

        # 1D wave spectrum E(f)     
        spec1d_freq = np.sum(np.multiply(spec2d,dtheta),axis=1)  
        
        # Getting the frequency band of maximum value of the 
        # frequency spectrum E(f)
        [fp_idx] = np.where(spec1d_freq == np.max(spec1d_freq))
        
        # 1D wave spectrum E(θ) evaluated over the peak frequency bin
        spec1d_dir_fp = spec2d[fp_idx[0],:]*df[fp_idx[0]]
        
        # Calculating the mean direction over the E(θ)
        dp = mean_direction([], [], dir_vec, spec1d_dir_fp, False)
                
        return dp

######### Calculating the peak frequency (coordinate of maximum value) ########

def peak_frequency_max(freq_vec, dir_vec, spec, two_dim_flag):
    
    if two_dim_flag == True:
        
        # Checking the spectrum orientation
        spec2d = spec_orientation(freq_vec, dir_vec, spec)    
        
        # Calculate the peak frequency, period and wavelength    
        [fp_idx, dp_idx] = np.where(spec2d == np.max(spec2d))

        # Getting the peak parameters fp, tp, and lp
        fp = freq_vec[fp_idx[0]]
        tp = 1/fp
        lp = (tp**2)*1.56
        
        return fp, tp, lp  
        
    elif two_dim_flag == False:
        
        # 1D wave spectrum E(f) 
        spec1d_freq = spec

        # Calculate the peak frequency, period and wavelength    
        [fp_idx] = np.where(spec1d_freq == np.max(spec1d_freq))
 
        # Getting the peak parameters fp, tp, and lp
        fp = freq_vec[fp_idx[0]]
        tp = 1/fp
        lp = (tp**2)*1.56
        
        return fp, tp, lp  

######## Calculating the peak direction (coordinate of maximum value) #########

def peak_direction_max(freq_vec, dir_vec, spec, two_dim_flag):
    
    if two_dim_flag == True:
        
        # Checking the spectrum orientation
        spec2d = spec_orientation(freq_vec, dir_vec, spec)    
        
        # Calculate the peak direction
        [fp_idx, dp_idx] = np.where(spec2d == np.max(spec2d))
        dp = dir_vec[dp_idx[0]]
        
        return dp
        
    elif two_dim_flag == False:
        
        # 1D wave spectrum E(θ) 
        spec1d_dir = spec

        # Calculate the peak frequency, period and wavelength    
        [dp_idx] = np.where(spec1d_dir == np.max(spec1d_dir))
        dp = dir_vec[dp_idx[0]]
        
        return dp


    