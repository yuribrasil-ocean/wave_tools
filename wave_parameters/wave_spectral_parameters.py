#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:05:39 2023

Script/Function: wave_spectral_parameters.py

Author: Yuri Brasil

e-mail: yuri.brasil@oceanica.ufrj.br

Modification: September 01, 2025

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

################## Calculating the d vector (df or dk) ########################

def d_logarithmic(spec_vector):    
    
    """
    Computes the interval between adjacent bins of a logarithmically (or 
    non-uniformly) spaced spectral vector.
    
    The interval is calculated using centered differences for interior points 
    and half-width differences for the first and last bins.
    
    Parameters
    ----------
    spec_vector: ndarray - Frequency (Hz) or wavenumber (rad/m) vector.
    
    Returns
    -------
    d_vec: ndarray - Vector containing the width associated with each spectral 
           bin. This vector can be used for numerical integration of spectra, 
           for example:
                
            m0 = Σ E(f) Δf or m0 = Σ E(k) Δk.
            
    """
    
    # Creating an empty array and the logarithmic ratio
    d_vec = np.zeros(len(spec_vector))
    
    # Loop to calculate the df elements (except the first and the last ones)
    # delta_f = (f_i+1 - f_i-1)/2 or delta_k = (k_i+1 - k_i-1)/2
    for i in range(1,len(spec_vector)-1):
        d_vec[i] = (spec_vector[i+1]-spec_vector[i-1])/2

    # Calculating the first and last elements
    d_vec[0] = (spec_vector[1]-spec_vector[0])/2
    d_vec[-1] = (spec_vector[-1]-spec_vector[-2])/2

    return d_vec

########### Correct spectrum orientation (if needed) and nans #################

def check_spec(dir_vec, freq_vec, spec):    
    
    """
    Checks the integrity and orientation of a directional spectrum.

    The function replaces NaN values by zeros, verifies whether the spectrum 
    is empty, and ensures that its dimensions follow the convention E(θ,f) or
    E(θ,k).

    Parameters
    ----------
    dir_vec: ndarray - Direction vector (degrees).

    freq_vec: ndarray - Frequency (Hz) or wavenumber (rad/m) vector.

    spec: ndarray - Two-dimensional directional spectrum.

    Returns
    -------
    spec2d: ndarray - Directional spectrum with dimensions 
             (n_directions, n_frequencies).

    Raises
    ------
    ValueError: If the spectrum is empty or its dimensions are inconsistent.
    
    """
    
    # Handle null spectrum (all NaN or all zeros)
    if np.all(np.isnan(spec)) or np.nanmax(spec) == 0:
        raise ValueError("Spectrum is null (all NaN or all zeros).")        
    
    # Replace nan elements by zero
    spec = np.nan_to_num(spec, nan=0.0)      
    
    # Check the spectrum orientation and, if necessary, correct the spectrum 
    # orientation that should be E(θ,f) or E(θ,k)
    if spec.shape == (len(dir_vec),len(freq_vec)):
        spec2d = spec
    elif spec.shape == (len(freq_vec),len(dir_vec)):
        spec2d = np.transpose(spec) 
    else:
        raise ValueError("Input spectrum dimensions do not match frequency and direction vectors.")
    
    return spec2d
   
########################### AVERAGE PARAMETERS ################################

############## Calculating the Significant Wave Height (Hs) ###################

def hs_spec(dir_vec, f_or_k_vec, delta, spec, two_dim_flag):    
    
    """
    Calculates the significant wave height from from a wave spectrum 
    (Hm0 or Hs).
    
    Parameters
    ----------
    dir_vec: ndarray -  Direction vector (degrees).
    
    f_or_k_vec: ndarray - Frequency (Hz) or wavenumber (rad/m) vector.
    
    delta: ndarray - Frequency (Δf) or wavenumber (Δk) increments.
    
    spec: ndarray - Two-dimensional directional spectrum or one-dimensional 
           spectrum.
    
    two_dim_flag: bool
        True  -> spec is E(θ,f) or E(θ,k)
        False -> spec is E(f) or E(k)
    
    Returns
    -------
    hs: float - Significant wave height (m).
        
    """
             
    # Calculating the width of direction vector bin in radians
    dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])      
    
    # Checking the dimensions
    if two_dim_flag == True:
        
        # Checking the spectrum
        spec2d = check_spec(dir_vec, f_or_k_vec, spec)    
        
        # 1D wave spectrum E(f)     
        spec1d = np.sum(np.multiply(spec2d,dtheta),axis=0)  
        
        # m0
        m0 = np.sum(np.multiply(spec1d,delta))
        
        # Calculating the Hs    
        hs = 4*np.sqrt(m0)

    # Checking the dimensions        
    elif two_dim_flag == False:
        
        # 1d wave spectrum
        spec1d = spec
        
        # m0
        m0 = np.sum(np.multiply(spec1d,delta))
        
        # Calculating the Hs    
        hs = 4.01*np.sqrt(m0)
    
    return hs

######################### Calculating the mean frequency ######################

def mean_frequency(dir_vec, freq_vec, df, spec, two_dim_flag, tm_flag):
    
    """
    Calculates the mean spectral frequency, period and wavelength.
    
    Parameters
    ----------
    dir_vec: ndarray - Direction vector (degrees).
    
    freq_vec: ndarray - Frequency vector (Hz).
    
    df: ndarray - Frequency-bin widths.
    
    spec: ndarray - Two-dimensional directional spectrum or one-dimensional 
          spectrum.
    
    two_dim_flag : bool
        True  -> spec is E(θ,f)
        False -> spec is E(f)
    
    tm_flag: int - Mean period definition.
    
        -1: Tm-1 = m-1/m0    
         1: Tm01 = m0/m1    
         2: Tm02 = sqrt(m0/m2)
    
    Returns
    -------
    tm: float - Mean wave period (s).
    
    fm: float - Mean frequency (Hz).
    
    lm: float - Deep-water mean wavelength (m).
    
    """

    if two_dim_flag == True:
        
        # Checking the spectrum
        spec2d = check_spec(dir_vec, freq_vec, spec)    

        # Calculate the dtheta element
        dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])

        # 1D wave spectrum E(f)     
        spec1d_freq = np.sum(np.multiply(spec2d,dtheta),axis=0)
                
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

def mean_direction(dir_vec, freq_vec, df, spec, two_dim_flag):
    
    """
    Calculates the mean wave direction using the first directional moment.
    
    Parameters
    ----------
    dir_vec: ndarray - Direction vector (degrees).
    
    freq_vec: ndarray - Frequency vector (Hz).
    
    df: ndarray - Frequency-bin widths.
    
    spec: ndarray - Two-dimensional directional spectrum or one-dimensional 
           directional distribution.
    
    two_dim_flag: bool
                  True  -> spec is E(θ,f)    
                  False -> spec is E(θ)
    
    Returns
    -------
    dm: float - Mean wave direction (degrees, 0-360).
    
    """

    if two_dim_flag == True:

        # Checking the spectrum
        spec2d = check_spec(dir_vec, freq_vec, spec)   

        # Calculating the dtheta element
        dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])
        radians_vec = np.deg2rad(dir_vec)
       
        # 1D wave spectrum E(θ)         
        spec1d_dir = np.sum(np.multiply(spec2d,df[np.newaxis,:]), axis=1)

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

def peak_frequency(dir_vec, freq_vec, spec, two_dim_flag):
    
    """
    Calculates the peak frequency by quadratic interpolation in logarithmic 
    frequency space.

    Parameters
    ----------
    dir_vec: ndarray - Direction vector (degrees).

    freq_vec: ndarray - Frequency vector (Hz). Assumed logarithmically spaced.

    spec: ndarray - 2D directional spectrum or 1D frequency spectrum.

    two_dim_flag: bool
                  True  -> spec is E(f,θ)
                  False -> spec is E(f)

    Returns
    -------
    fp: float - Peak frequency (Hz)
    tp: float - Peak period (s)
    lp: float - Deep-water wavelength (m)
    
    """

    ######################## Obtain the 1D spectrum ########################

    if two_dim_flag:

        # Checking the spectrum
        spec2d = check_spec(dir_vec, freq_vec, spec)

         # Handle null spectrum (all NaN or all zeros)
        if np.all(np.isnan(spec2d)) or np.nanmax(spec2d) == 0:
            raise ValueError("Spectrum is null (all NaN or all zeros).")

        # Direction increment
        dtheta = np.deg2rad(dir_vec[1] - dir_vec[0])

        # 1D wave spectrum E(f)     
        spec1d_freq = np.sum(spec2d * dtheta, axis=0)

    else:

        # 1D wave spectrum E(f)     
        spec1d_freq = np.asarray(spec)

        # Handle null spectrum (all NaN or all zeros)
        if np.all(np.isnan(spec1d_freq)) or np.nanmax(spec1d_freq) == 0:
            raise ValueError("Spectrum is null (all NaN or all zeros).")

    # Locate discrete peak
    fp_idx = np.nanargmax(spec1d_freq)

    ######################## Boundary case ########################

    # If the maximum occurs at the edge,
    # interpolation is impossible.
    if fp_idx == 0 or fp_idx == len(freq_vec) - 1:

        # Peak frequency and derived parameters
        fp = freq_vec[fp_idx]
        tp = 1/fp
        lp = (tp**2)*1.56

        # Getting the peak parameters fp, tp, and lp
        return fp, tp, lp

    ######################## Three-point quadratic fit ########################

    # Three neighboring frequencies
    # x = np.log(freq_vec[fp_idx-1:fp_idx+2])
    x = np.array([-1.0, 0.0, 1.0])
    
    # Three neighboring spectral values
    y = spec1d_freq[fp_idx-1:fp_idx+2]
    
    # Bin coordinates
    x = np.array([-1.0, 0.0, 1.0])
    
    # Quadratic fit
    a, b, c = np.polyfit(x, y, 2)
    
    # If the parabola is nearly flat or opens upward,
    # fall back to the discrete peak.
    if np.isclose(a, 0.0) or a >= 0:
    
        # Peak frequency
        fp = freq_vec[fp_idx]
    
    else:
    
        # Vertex of the parabola
        delta = -b / (2 * a)
    
        # Keep the solution inside the neighboring bins
        delta = np.clip(delta, -1.0, 1.0)
    
        # Increment
        gamma = freq_vec[1]/freq_vec[0]
    
        # Peak frequency
        fp = freq_vec[fp_idx]*gamma**delta

    # Derived parameters
    tp = 1 / fp
    lp = 1.56 * tp**2

    return fp, tp, lp

######################### Calculating the peak direction ######################

def peak_direction(dir_vec, freq_vec, df, spec):
    
    
    """
    Calculates the peak wave direction.
    
    The peak direction is defined as the mean direction of the directional
    distribution corresponding to the peak-frequency bin.
    
    Parameters
    ----------
    dir_vec: ndarray - Direction vector (degrees).
    
    freq_vec: ndarray - Frequency vector (Hz).
    
    df: ndarray - Frequency-bin widths.
    
    spec: ndarray - Two-dimensional directional spectrum.
    
    Returns
    -------
    
    dp: float - Peak wave direction (degrees).
    
    """

    # Checking the spectrum
    spec2d = check_spec(dir_vec, freq_vec, spec)   

    # Calculate the dtheta element
    dtheta = np.deg2rad(dir_vec[1]-dir_vec[0])

    # 1D wave spectrum E(f)     
    spec1d_freq = np.sum(np.multiply(spec2d,dtheta),axis=0)  
    
    # Getting the frequency band of maximum value of the 
    # frequency spectrum E(f)
    [fp_idx] = np.where(spec1d_freq == np.max(spec1d_freq))
    
    # 1D wave spectrum E(θ) evaluated over the peak frequency bin
    spec1d_dir_fp = spec2d[:,fp_idx[0]]*df[fp_idx[0]]
    
    # Calculating the mean direction over the E(θ)
    dp = mean_direction(dir_vec, [], [], spec1d_dir_fp, False)
            
    return dp

######### Calculating the peak frequency (coordinate of maximum value) ########

def peak_frequency_max(dir_vec, freq_vec, spec, two_dim_flag):
    
    """
    Calculates the peak frequency from the maximum value of the spectrum.
    Unlike peak_frequency(), no interpolation is performed.
    
    Parameters
    ----------
    dir_vec: ndarray - Direction vector (degrees).
    
    freq_vec: ndarray - Frequency vector (Hz).
    
    spec: ndarray - Two-dimensional directional spectrum or one-dimensional 
           spectrum.
    
    two_dim_flag: bool
                  True  -> spec is E(θ,f)    
                  False -> spec is E(f)
    
    Returns
    -------
    fp: float - Peak frequency (Hz).
    
    tp: float - Peak period (s).
    
    lp: float - Deep-water peak wavelength (m).
    
    """
    
    if two_dim_flag == True:
        
        # Checking the spectrum
        spec2d = check_spec(dir_vec, freq_vec, spec)    
        
        # Calculate the peak frequency, period and wavelength
        [dp_idx, fp_idx] = np.where(spec2d == np.max(spec2d))

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

def peak_direction_max(dir_vec, freq_vec, spec, two_dim_flag):
    
    """
    Calculates the peak direction from the maximum value of the spectrum.
    Unlike peak_direction(), no directional averaging is performed.

    Parameters
    ----------
    dir_vec: ndarray - Direction vector (degrees).

    freq_vec: ndarray - Frequency vector (Hz).

    spec: ndarray - Two-dimensional directional spectrum or one-dimensional directional
          distribution.

    two_dim_flag: bool
                  True  -> spec is E(θ,f)    
                  False -> spec is E(θ)
    
    Returns
    -------
    dp: float - Peak wave direction (degrees).
    
    """
    
    if two_dim_flag == True:
        
        # Checking the spectrum
        spec2d = check_spec(dir_vec, freq_vec, spec)           

        # Calculate the peak direction
        [dp_idx, fp_idx] = np.where(spec2d == np.max(spec2d))
        dp = dir_vec[dp_idx[0]]
        
        return dp
        
    elif two_dim_flag == False:
        
        # 1D wave spectrum E(θ) 
        spec1d_dir = spec

        # Calculate the peak frequency, period and wavelength    
        [dp_idx] = np.where(spec1d_dir == np.max(spec1d_dir))
        dp = dir_vec[dp_idx[0]]
        
        return dp

