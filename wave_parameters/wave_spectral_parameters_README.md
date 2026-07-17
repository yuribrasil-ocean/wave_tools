# 🌊 Wave Spectral Parameters

This Python module provides functions to compute average and peak wave parameters from directional wave spectra, including:

- Significant Wave Height (Hs or SWH)
- Mean Wave Period (Tm), Frequency (Fm), and Wavelength (Lm)
- Peak Period (Tp), Frequency (Fp), and Wavelength (Lp)
- Mean and Peak Wave Directions (Dm and Dp)
- Spectral conversions between frequency–direction and wavenumber–direction domains

📁 **Script**: `wave_spectral_parameters.py`  
👤 **Author**: Yuri Brasil  
📧 **Email**: yuri.brasil@oceanica.ufrj.br  
📅 **Last Modified**: June 22, 2025

---

## 🧩 Features

- ✅ Works with 1D and 2D spectral data (frequency × direction)
- ✅ Supports nonuniform and logarithmic frequency spacing
- ✅ Preserves spectral energy during conversion between domains
- ✅ Provides both analytical and parabolic-approximation methods for peak extraction

---

## 📚 How to Use

1. **Compute frequency/wavenumber bin spacing:**

```python
df = d_logarithmic(freq_vec)  # or use d_logarithmic_old(freq_vec, increment)

spec_k, k_vec, dir_vec = wave_spec_conversion(freq_vec, dir_vec, df, spec_2d, 'to_k')
spec_f, f_vec, dir_vec = wave_spec_conversion(k_vec, dir_vec, dk, spec_k, 'to_f')

hs = hs_spec(freq_vec, df, dir_vec, spec_2d, True)
tm, fm, lm = mean_frequency(freq_vec, df, dir_vec, spec_2d, True, tm_flag=1)
dm = mean_direction(freq_vec, df, dir_vec, spec_2d, True)

fp, tp, lp = peak_frequency(freq_vec, dir_vec, spec_2d, True)
dp = peak_direction(freq_vec, df, dir_vec, spec_2d)

# Or using the raw maximum:
fp_max, tp_max, lp_max = peak_frequency_max(freq_vec, dir_vec, spec_2d, True)
dp_max = peak_direction_max(freq_vec, dir_vec, spec_2d, True)


For questions, suggestions, or bug reports, contact:

Yuri Brasil
📧 yuri.brasil@oceanica.ufrj.br
🌐 https://github.com/yuribrasil-ocean