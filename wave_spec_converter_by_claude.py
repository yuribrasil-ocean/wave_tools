#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 12:16:59 2025

@author: yuri
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def interpolate_directional_spectrum_x(freq_vec, dir_vec, wave_spectrum,
                                       freq_vec_new, dir_vec_new, 
                                       log_freq_vec=True, 
                                       interp_method='linear'):
    """
    Interpolate a directional wave spectrum to a new frequency-direction grid.
    
    Parameters:
    -----------
    spectrum : array_like, shape (n_freq, n_dir)
        Original directional spectrum values
    freq_orig : array_like, shape (n_freq,)
        Original frequency vector
    dir_orig : array_like, shape (n_dir,)
        Original direction vector (degrees)
    freq_new : array_like, shape (m_freq,)
        Target frequency vector
    dir_new : array_like, shape (m_dir,)
        Target direction vector (degrees)
    log_freq : bool, default=True
        Whether to interpolate in log-frequency space
    method : str, default='linear'
        Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns:
    --------
    spectrum_interp : ndarray, shape (m_freq, m_dir)
        Interpolated directional spectrum
    """
    
    # Handle direction wrapping (0° to 360°)
    dir_vec = np.array(dir_vec) % 360
    dir_vec_new = np.array(dir_vec_new) % 360
    
    # Loop convert into log interpolation in log space if requested
    if log_freq_vec:
        freq_coord_orig = np.log(freq_orig)
        freq_coord_new = np.log(freq_new)
    else:
        freq_coord_orig = freq_orig
        freq_coord_new = freq_new
    
    # Create the interpolator
    interpolator = RegularGridInterpolator((freq_coord_orig, dir_vec),
                                           wave_spectrum, method=interp_method,
                                           bounds_error=False, fill_value=0.0)

    # Create new coordinate grids
    freq_grid, dir_grid = np.meshgrid(freq_coord_new, dir_vec_new, indexing='ij')
    
    # Interpolate
    interpolated_spectrum = interpolator((freq_grid, dir_grid))
    
    return interpolated_spectrum

def create_sample_era5_spectrum():
    """Create a sample ERA5-like directional spectrum for testing."""
    
    # Original ERA5-like grid
    freq_orig = np.logspace(-1, 0.5, 30)  # 30 frequencies, log-spaced
    dir_orig = np.linspace(0, 360, 36, endpoint=False)  # 36 directions
    
    # Create a realistic-looking spectrum
    # Peak frequency around 0.1 Hz, dominant direction around 270°
    F, D = np.meshgrid(freq_orig, dir_orig, indexing='ij')
    
    # JONSWAP-like frequency spectrum
    fp = 0.1  # peak frequency
    gamma = 3.3
    alpha = 0.0081
    
    # Frequency part (JONSWAP)
    freq_spectrum = alpha * (F**-5) * np.exp(-1.25 * (fp/F)**4)
    sigma = np.where(F <= fp, 0.07, 0.09)
    freq_spectrum *= gamma ** np.exp(-0.5 * ((F - fp) / (sigma * fp))**2)
    
    # Directional spreading (cos^2s distribution)
    theta_mean = 270  # degrees
    s = 10  # spreading parameter
    dir_spread = np.cos(np.pi * (D - theta_mean) / 180)**s
    dir_spread = np.maximum(dir_spread, 0)
    
    # Combine frequency and directional components
    spectrum = freq_spectrum * dir_spread
    
    return spectrum, freq_orig, dir_orig

# def interpolate_directional_spectrum(spectrum, freq_orig, dir_orig, freq_new, dir_new, 
#                                    log_freq=True, method='linear'):
#     """
#     Interpolate a directional wave spectrum to a new frequency-direction grid.
    
#     Parameters:
#     -----------
#     spectrum : array_like, shape (n_freq, n_dir)
#         Original directional spectrum values
#     freq_orig : array_like, shape (n_freq,)
#         Original frequency vector
#     dir_orig : array_like, shape (n_dir,)
#         Original direction vector (degrees)
#     freq_new : array_like, shape (m_freq,)
#         Target frequency vector
#     dir_new : array_like, shape (m_dir,)
#         Target direction vector (degrees)
#     log_freq : bool, default=True
#         Whether to interpolate in log-frequency space
#     method : str, default='linear'
#         Interpolation method ('linear', 'nearest', 'cubic')
    
#     Returns:
#     --------
#     spectrum_interp : ndarray, shape (m_freq, m_dir)
#         Interpolated directional spectrum
#     """
    
#     # Handle direction wrapping (0-360 degrees)
#     dir_orig = np.array(dir_orig) % 360
#     dir_new = np.array(dir_new) % 360
    
#     # For frequency interpolation in log space if requested
#     if log_freq:
#         freq_coord_orig = np.log(freq_orig)
#         freq_coord_new = np.log(freq_new)
#     else:
#         freq_coord_orig = freq_orig
#         freq_coord_new = freq_new
    
#     # Create the interpolator
#     interpolator = RegularGridInterpolator(
#         (freq_coord_orig, dir_orig), 
#         spectrum, 
#         method=method,
#         bounds_error=False,
#         fill_value=0.0
#     )
    
#     # Create new coordinate grids
#     freq_grid, dir_grid = np.meshgrid(freq_coord_new, dir_new, indexing='ij')
    
#     # Interpolate
#     spectrum_interp = interpolator((freq_grid, dir_grid))
    
#     return spectrum_interp

def create_sample_era5_spectrum():
    """Create a sample ERA5-like directional spectrum for testing."""
    
    # Original ERA5-like grid
    freq_orig = np.logspace(-1, 0.5, 30)  # 30 frequencies, log-spaced
    dir_orig = np.linspace(0, 360, 36, endpoint=False)  # 36 directions
    
    # Create a realistic-looking spectrum
    # Peak frequency around 0.1 Hz, dominant direction around 270°
    F, D = np.meshgrid(freq_orig, dir_orig, indexing='ij')
    
    # JONSWAP-like frequency spectrum
    fp = 0.1  # peak frequency
    gamma = 3.3
    alpha = 0.0081
    
    # Frequency part (JONSWAP)
    freq_spectrum = alpha * (F**-5) * np.exp(-1.25 * (fp/F)**4)
    sigma = np.where(F <= fp, 0.07, 0.09)
    freq_spectrum *= gamma ** np.exp(-0.5 * ((F - fp) / (sigma * fp))**2)
    
    # Directional spreading (cos^2s distribution)
    theta_mean = 270  # degrees
    s = 10  # spreading parameter
    dir_spread = np.cos(np.pi * (D - theta_mean) / 180)**s
    dir_spread = np.maximum(dir_spread, 0)
    
    # Combine frequency and directional components
    spectrum = freq_spectrum * dir_spread
    
    return spectrum, freq_orig, dir_orig

def plot_spectra_comparison(spectrum_orig, freq_orig, dir_orig, 
                          spectrum_interp, freq_new, dir_new):
    """Plot original and interpolated spectra for comparison."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original spectrum
    F1, D1 = np.meshgrid(freq_orig, dir_orig, indexing='ij')
    im1 = ax1.contourf(D1, F1, spectrum_orig, levels=20, cmap='viridis')
    ax1.set_xlabel('Direction (degrees)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Original Spectrum ({spectrum_orig.shape[0]}x{spectrum_orig.shape[1]})')
    ax1.set_yscale('log')
    plt.colorbar(im1, ax=ax1, label='Spectral Density')
    
    # Interpolated spectrum
    F2, D2 = np.meshgrid(freq_new, dir_new, indexing='ij')
    im2 = ax2.contourf(D2, F2, spectrum_interp, levels=20, cmap='viridis')
    ax2.set_xlabel('Direction (degrees)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(f'Interpolated Spectrum ({spectrum_interp.shape[0]}x{spectrum_interp.shape[1]})')
    ax2.set_yscale('log')
    plt.colorbar(im2, ax=ax2, label='Spectral Density')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample ERA5 spectrum (30 freq x 36 dir)
    spectrum_orig, freq_orig, dir_orig = create_sample_era5_spectrum()
    
    print(f"Original spectrum shape: {spectrum_orig.shape}")
    print(f"Original frequency range: {freq_orig[0]:.3f} - {freq_orig[-1]:.3f} Hz")
    print(f"Original direction range: {dir_orig[0]:.1f} - {dir_orig[-1]:.1f} degrees")
    
    # Define new target grid (25 freq x 24 dir)
    freq_new = np.logspace(np.log10(freq_orig[0]), np.log10(freq_orig[-1]), 25)
    dir_new = np.linspace(0, 360, 24, endpoint=False)
    
    print(f"\nTarget frequency range: {freq_new[0]:.3f} - {freq_new[-1]:.3f} Hz")
    print(f"Target direction range: {dir_new[0]:.1f} - {dir_new[-1]:.1f} degrees")
    
    # Interpolate spectrum
    # spectrum_interp = interpolate_directional_spectrum_x(
    #     spectrum_orig, freq_orig, dir_orig, freq_new, dir_new,
    #     log_freq_vec=True, 
    #     interp_method='linear'
    # )
    
    spectrum_interp = interpolate_directional_spectrum_x(
        freq_orig, dir_orig, spectrum_orig, freq_new, dir_new,
        log_freq_vec=True, 
        interp_method='linear'
    )
    
    # def interpolate_directional_spectrum_x(freq_vec, dir_vec, wave_spectrum,
    #                                        freq_vec_new, dir_vec_new, 
    #                                        log_freq_vec=True, 
    #                                        interp_method='linear'):
    
    print(f"\nInterpolated spectrum shape: {spectrum_interp.shape}")
    
    # Compare total energy conservation
    total_energy_orig = np.trapz(np.trapz(spectrum_orig, dir_orig), freq_orig)
    total_energy_interp = np.trapz(np.trapz(spectrum_interp, dir_new), freq_new)
    energy_ratio = total_energy_interp / total_energy_orig
    
    print(f"\nEnergy conservation check:")
    print(f"Original total energy: {total_energy_orig:.6f}")
    print(f"Interpolated total energy: {total_energy_interp:.6f}")
    print(f"Energy ratio: {energy_ratio:.4f}")
    
    # Plot comparison
    plot_spectra_comparison(spectrum_orig, freq_orig, dir_orig,
                          spectrum_interp, freq_new, dir_new)
    
    # Example: Extract 1D frequency spectrum (integrated over directions)
    freq_spectrum_orig = np.trapz(spectrum_orig, dir_orig, axis=1)
    freq_spectrum_interp = np.trapz(spectrum_interp, dir_new, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(freq_orig, freq_spectrum_orig, 'o-', label='Original', alpha=0.7)
    plt.loglog(freq_new, freq_spectrum_interp, 's-', label='Interpolated', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency Spectrum')
    plt.title('1D Frequency Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()