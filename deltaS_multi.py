import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed, parallel_backend, cpu_count


# Coefficients for metallicity calculation, line 1 in table 8 Crestani+2021
C0, C1, C2, C3, C4 = -3.84323, 0.36828, -0.22182, 0.00433, 0.51481

# Crestani+2021 wavelenght (in AA) boundaries for continuum and line
ca_boundary = np.array([3910.00, 3925.00, 3940.00, 3955.00, 3923.67, 3943.67])
hd_boundary = np.array([4010.00, 4060.00, 4145.00, 4195.00, 4091.74, 4111.74])
hg_boundary = np.array([4230.00, 4280.00, 4400.00, 4450.00, 4330.47, 4350.47])
hb_boundary = np.array([4750.00, 4800.00, 4920.00, 4970.00, 4851.33, 4871.33])

# Wavelenghts (in AA) of spectral lines
line_centers = np.array([3933.66, 4101.75, 4340.47, 4861.34])


# Starting constants
NUMBER_OF_ITERATIONS = 3
NUMBER_OF_THREADS = cpu_count()


def center_line(wave, flux, eflux, deg=4):
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

    if (flux.size == wave.size == eflux.size) & (wave.size > 0 ):

        x_fit = np.linspace(min(wave), max(wave), 1000)
        
        f_spline = InterpolatedUnivariateSpline(wave, flux, w=eflux**-2, k=2)
        wave_offset = x_fit[np.argmin(f_spline(x_fit))]

        absolute_diff_in_line = abs(line_centers - wave_offset)
        slicer_shift = np.where(absolute_diff_in_line == min(absolute_diff_in_line))
        
        wave_shift = line_centers[slicer_shift] - wave_offset
        
        return wave_shift[0]

    else:

        return None


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




def multi_proc_function(wave, flux_r, eflux, boundary, select_region, wave_shift):
    """
    Implementation of the least squares method.

    Parameters
    ----------
    wave : 'array_like, shape (N, )'
        Wavelenght in angstroms 
    flux_r : 'array_like, shape (N, )'
        Flux in any given unit
    eflux : 'array_like, shape (N, )'
        Uncertainties on flux
    boundary : 'array_like, shape (4, 6)'
        Boundaries for line analysis
    select_region : 'array_like, shape (4, M)'
        Regions for continuum normalization
    wave_shift : 'array_like, shape (4, )'
        Shifts in wavelenght for individual line

    Returns:
    ------- 
    equivalent_width : 'float'
        Pseudo-equivalent width of individual line
    snr : 'float'
        Signal-noise-ratio for a given spectral line.
    """     


    number_of_lines = len(select_region)

    equivalent_width, snr = np.empty((2, number_of_lines))


    for k, (boun, reg, shif) in enumerate(zip(boundary, select_region, wave_shift)):

        wavelenght_shifted = wave[reg] + shif

        flux_norm, eflux_norm, snr[k] = normalize_flux(wavelenght_shifted, 
                                                       flux_r[reg], 
                                                       eflux[reg], 
                                                       boun)

        equivalent_width[k] = calculate_rr_ew(wavelenght_shifted, 
                                              flux_norm, 
                                              eflux_norm, 
                                              boun)

    return equivalent_width, snr

# ----------------------------------------------------------------------------

def est_individual_lines(wave, flux, eflux, boundary, select_region, wave_shift):
    """
    Implementation of the least squares method.

    Parameters
    ----------
    wave : 'array_like, shape (N, )'
        Wavelenght in angstroms 
    flux_r : 'array_like, shape (N, )'
        Flux in any given unit
    eflux : 'array_like, shape (N, )'
        Uncertainties on flux
    boundary : 'array_like, shape (4, 6)'
        Boundaries for line analysis
    select_region : 'array_like, shape (4, M)'
        Regions for continuum normalization
    wave_shift : 'array_like, shape (4, )'
        Shifts in wavelenght for individual line

    Returns:
    ------- 
    ew_for_all_lines : 'array_like, shape (NUMBER_OF_ITERATIONS, 4)'
        Pseudo-equivalent widths of all lines.
    snr_for_all_lines : 'array_like, shape (NUMBER_OF_ITERATIONS, 4)'
        Signal-noise-ratio for all spectral lines.
    """     
    
    flux_r = np.array([np.random.normal(flux, eflux) 
                      for i in range(NUMBER_OF_ITERATIONS)])


    with parallel_backend('multiprocessing'):
        res = Parallel(n_jobs=NUMBER_OF_THREADS)((delayed(multi_proc_function))
            (wave, abs(flux_r[i]), eflux, boundary, select_region, wave_shift) 
            for i in range(NUMBER_OF_ITERATIONS))


    res = np.array(res)

    ew_for_all_lines = res[:,0]
    snr_for_all_lines = res[:,1]


    return ew_for_all_lines, snr_for_all_lines


# ----------------------------------------------------------------------------






def main():


    data = np.genfromtxt("/home/zdenek/Dropbox/___CATS/---+++/deltaS/2880057803597178880.corr")
    data = np.genfromtxt("/home/zdenek/tools/iSpec/2599745565482838016-corr.txt")
    
    remove_bad_values = np.where( (~np.isnan(data[:,0]) & np.isfinite(data[:,0])) &
                                  (~np.isnan(data[:,1]) & np.isfinite(data[:,1])) & 
                                  (~np.isnan(data[:,2]) & np.isfinite(data[:,2])) &
                                  (data[:,1] / data[:,2] > 2.) )

    wave, flux, eflux = (data[:,0][remove_bad_values], 
                         data[:,1][remove_bad_values], 
                         data[:,2][remove_bad_values])


    # Selecting portions of the spectra to get important sections
    ca_region = np.where((wave > min(ca_boundary)) & (wave < max(ca_boundary)))
    hd_region = np.where((wave > min(hd_boundary)) & (wave < max(hd_boundary)))
    hg_region = np.where((wave > min(hg_boundary)) & (wave < max(hg_boundary)))
    hb_region = np.where((wave > min(hb_boundary)) & (wave < max(hb_boundary)))


    mask_select_line_ca = np.where((wave > ca_boundary[4]) 
                                 & (wave < ca_boundary[5]))
    wave_shift_ca = center_line(wave[mask_select_line_ca], 
                                flux[mask_select_line_ca], 
                                eflux[mask_select_line_ca])

    mask_select_line_hd = np.where((wave > hd_boundary[4]) 
                                 & (wave < hd_boundary[5]))
    wave_shift_hd = center_line(wave[mask_select_line_hd], 
                                flux[mask_select_line_hd], 
                                eflux[mask_select_line_hd])

    mask_select_line_hg = np.where((wave > hg_boundary[4]) 
                                 & (wave < hg_boundary[5]))
    wave_shift_hg = center_line(wave[mask_select_line_hg], 
                                flux[mask_select_line_hg], 
                                eflux[mask_select_line_hg])

    mask_select_line_hb = np.where((wave > hb_boundary[4]) 
                                 & (wave < hb_boundary[5]))
    wave_shift_hb = center_line(wave[mask_select_line_hb], 
                                flux[mask_select_line_hb], 
                                eflux[mask_select_line_hb])


    shifts = [wave_shift_ca, wave_shift_hd, wave_shift_hg, wave_shift_hb]
    boundaries = [ca_boundary, hd_boundary, hg_boundary, hb_boundary]
    region = [ca_region, hd_region, hg_region, hb_region]

    ew_ls, signal_to_noise = est_individual_lines(wave, flux, eflux, 
                                                  boundaries, region, shifts)


    feh = C0 + C1*ew_ls[:,0] + C2*ew_ls[:,1] + C3*ew_ls[:,2] + C4*ew_ls[:,3]
        

    print ("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f " %(
        np.nanmean(feh), np.nanstd(feh), 
        np.nanmean(ew_ls[:,0]), np.nanstd(ew_ls[:,0]), 
        np.nanmean(ew_ls[:,1]), np.nanstd(ew_ls[:,1]), 
        np.nanmean(ew_ls[:,2]), np.nanstd(ew_ls[:,2]), 
        np.nanmean(ew_ls[:,3]), np.nanstd(ew_ls[:,3])))


    return None



if __name__ == "__main__":
    main()


