import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import time

# Coefficients for metallicity calculation, line 1 in table 8 Crestani+2021
C0, C1, C2, C3, C4 = -3.84323, 0.36828, -0.22182, 0.00433, 0.51481
NUMBER_OF_ITERATIONS = 10000

# Crestani+2021 wavelenght (in AA) boundaries for continuum and line
ca_boundary = np.array([3910.00, 3925.00, 3940.00, 3955.00, 3923.67, 3943.67])
hd_boundary = np.array([4010.00, 4060.00, 4145.00, 4195.00, 4091.74, 4111.74])
hg_boundary = np.array([4230.00, 4280.00, 4400.00, 4450.00, 4330.47, 4350.47])
hb_boundary = np.array([4750.00, 4800.00, 4920.00, 4970.00, 4851.33, 4871.33])

# Wavelenghts (in AA) of spectral lines
line_centers = np.array([3933.66, 4101.75, 4340.47, 4861.34])


def center_line(wave, flux, eflux, deg):
    """
    Re-centers the line after 
    line-of-sight velocity correction

    Parameters
    ----------
    wave : 'array_like, shape (N, )'
        Wavelenght in angstroms 
    flux : 'array_like, shape (N, )'
        Flux in any given unit
    eflux : 'array_like, shape (N, )'
        Uncertainties on flux
    deg : 'int'
        Degree of the polynomial fit

    Returns:
    ------- 
    wave_shift : 'float'
        Wavelenght shift to center 
        the spectral line.
    """ 

    x_fit = np.linspace(min(wave), max(wave), 1000)
    
    b_fit = np.polyfit(wave, flux, w=eflux**-2, deg=deg)
    
    wave_offset = x_fit[np.argmin(np.polyval(b_fit, x_fit))]
    
    absolute_diff_in_line = abs(line_centers - wave_offset)
    slicer_shift = np.where(absolute_diff_in_line == min(absolute_diff_in_line))
    
    wave_shift = line_centers[slicer_shift] - wave_offset
    
    return wave_shift


def normalize_flux(wave, flux, eflux, slicing_wave):
    """
    Flux normalization using regions from Crestani+2021

    Parameters
    ----------
    wave : 'array_like, shape (N, )'
        Wavelenght in angstroms 
    flux : 'array_like, shape (N, )'
        Flux in any given unit
    eflux : 'array_like, shape (N, )'
        Uncertainties on flux
    slicing_wave : 'array_like, shape (N, )'
        Boundaries for continuum determination

    Returns:
    ------- 
    flux_flip : 'array_like, shape (N, )'
        Normalized flux
    conti_err : 'array_like, shape (N, )'
        Error on the normalized flux
    snr : 'float'
        Signal-noise-ratio for given 
        spectral line.
    """     

    mask_select_conti = np.where( 
        ((wave > slicing_wave[0]) & (wave < slicing_wave[1])) 
        | ((wave > slicing_wave[2]) & (wave < slicing_wave[3])))

    poly_coef = np.polyfit(
        wave[mask_select_conti], 
        flux[mask_select_conti], 
        w=eflux[mask_select_conti]**-2, 
        deg=1)


    fitted_flux = np.polyval(poly_coef, wave)
    conti_flux = 1.0 + (flux - fitted_flux) / fitted_flux
    conti_err = (1.0 + (flux + eflux - fitted_flux) / fitted_flux) - conti_flux

    flux_flip = -1.0*conti_flux + 1.0
    flux_flip[flux_flip < 0.0] = 0.0

    snr = 1.0/np.std(flux_flip[mask_select_conti])

    return flux_flip, conti_err, snr


def calculate_rr_ew(wave, flux_flip, conti_err, slicing):
    """
    Implementation of the least squares method.

    Parameters
    ----------
    wave : 'array_like, shape (N, )'
        Wavelenght in angstroms 
    flux : 'array_like, shape (N, )'
        Flux in any given unit
    eflux : 'array_like, shape (N, )'
        Uncertainties on flux
    slicing : 'array_like, shape (N, )'
        Boundaries for EW determination

    Returns:
    ------- 
    pseudo_ew : 'float'
        Pseudo-equivalent width
    """     
    
    f = InterpolatedUnivariateSpline(wave, flux_flip, w=conti_err**-2, k=1)
    pseudo_ew = f.integral(slicing[4], slicing[5])
    
    return pseudo_ew



def main():
    start = time.time()
    # ----------------------------------------------------------------------------------------
    data = np.genfromtxt("/home/zdenek/Dropbox/___CATS/---+++/deltaS/2880057803597178880.corr")
    data = np.genfromtxt("/home/zdenek/tools/iSpec/2599745565482838016-corr.txt")

    wave, flux, eflux = data[:,0], data[:,1], data[:,2]


    # Selecting portions of the spectra to get important sections
    ca_region = np.where((wave > min(ca_boundary)) & (wave < max(ca_boundary)))
    hd_region = np.where((wave > min(hd_boundary)) & (wave < max(hd_boundary)))
    hg_region = np.where((wave > min(hg_boundary)) & (wave < max(hg_boundary)))
    hb_region = np.where((wave > min(hb_boundary)) & (wave < max(hb_boundary)))

    ew_ca, ew_hd, ew_hg, ew_hb, feh = np.empty((5, NUMBER_OF_ITERATIONS))
    snr_ca, snr_hd, snr_hg, snr_hb = np.empty((4, NUMBER_OF_ITERATIONS))

    mask_select_line_ca = np.where((wave > ca_boundary[4]) 
                                 & (wave < ca_boundary[5]))
    wave_shift_ca = center_line(wave[mask_select_line_ca], 
                                flux[mask_select_line_ca], 
                                eflux[mask_select_line_ca], 
                                deg=4)

    mask_select_line_hd = np.where((wave > hd_boundary[4]) 
                                 & (wave < hd_boundary[5]))
    wave_shift_hd = center_line(wave[mask_select_line_hd], 
                                flux[mask_select_line_hd], 
                                eflux[mask_select_line_hd], 
                                deg=4)

    mask_select_line_hg = np.where((wave > hg_boundary[4]) 
                                 & (wave < hg_boundary[5]))
    wave_shift_hg = center_line(wave[mask_select_line_hg], 
                                flux[mask_select_line_hg], 
                                eflux[mask_select_line_hg], 
                                deg=4)

    mask_select_line_hb = np.where((wave > hb_boundary[4]) 
                                 & (wave < hb_boundary[5]))
    wave_shift_hb = center_line(wave[mask_select_line_hb], 
                                flux[mask_select_line_hb], 
                                eflux[mask_select_line_hb], 
                                deg=4)




    for i in range(NUMBER_OF_ITERATIONS):

        flux_r = np.random.normal(flux, eflux)


        flux_norm, eflux_norm, snr_ca[i] = normalize_flux(wave[ca_region]+wave_shift_ca, 
                                                          flux_r[ca_region], 
                                                          eflux[ca_region], 
                                                          ca_boundary)
        ew_ca[i] = calculate_rr_ew(wave[ca_region]+wave_shift_ca, 
                                   flux_norm, 
                                   eflux_norm, 
                                   ca_boundary) 


        flux_norm, eflux_norm, snr_hd[i] = normalize_flux(wave[hd_region]+wave_shift_hd, 
                                                          flux_r[hd_region], 
                                                          eflux[hd_region], 
                                                          hd_boundary)
        ew_hd[i] = calculate_rr_ew(wave[hd_region]+wave_shift_hd, 
                                   flux_norm, 
                                   eflux_norm, 
                                   hd_boundary)


        flux_norm, eflux_norm, snr_hg[i] = normalize_flux(wave[hg_region]+wave_shift_hg, 
                                                          flux_r[hg_region], 
                                                          eflux[hg_region], 
                                                          hg_boundary)
        ew_hg[i] = calculate_rr_ew(wave[hg_region]+wave_shift_hg, 
                                   flux_norm, 
                                   eflux_norm, 
                                   hg_boundary)
        

        flux_norm, eflux_norm, snr_hb[i] = normalize_flux(wave[hb_region]+wave_shift_hb, 
                                                          flux_r[hb_region], 
                                                          eflux[hb_region], 
                                                          hb_boundary)
        ew_hb[i] = calculate_rr_ew(wave[hb_region]+wave_shift_hb, 
                                   flux_norm, 
                                   eflux_norm, 
                                   hb_boundary)


        feh[i] = C0 + C1*ew_ca[i] + C2*ew_hd[i] + C3*ew_hg[i] + C4*ew_hb[i]
        

    print ("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f " %(np.nanmean(feh), np.nanstd(feh), 
        np.nanmean(ew_ca), np.nanstd(ew_ca), 
        np.nanmean(ew_hd), np.nanstd(ew_hd), 
        np.nanmean(ew_hg), np.nanstd(ew_hg), 
        np.nanmean(ew_hb), np.nanstd(ew_hb)))
    # ---------------------------------------------------------------------------------------
    print('It took', time.time()-start, 'seconds.')
    return None



if __name__ == "__main__":
    main()