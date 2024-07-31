import numpy as np
import numpy.ma as ma
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.io import ascii
#from joblib import Parallel, delayed, parallel_backend, cpu_count


# Coefficients for metallicity calculation, line 1 in table 8 Crestani+2021
C0, C1, C2, C3, C4 = -3.84323, 0.36828, -0.22182, 0.00433, 0.51481
C0a, C1a, C2a, C3a = -3.75381, 0.39014, -0.19997, 0.38916
C0b, C1b, C2b, C4b = -3.84160, 0.36798, -0.21936, 0.51676
C0c, C1c, C3c, C4c = -3.79074, 0.35889, -0.21997, 0.50469
C0d, C1d, C2d = -3.48130, 0.36105, 0.14403
C0e, C1e, C3e = -3.70799, 0.38127, 0.17973
C0f, C1f, C4f = -3.92067, 0.38194, 0.25898


# Crestani+2021 wavelenght (in AA) boundaries for continuum and line
ca_boundary = np.array([3910.00, 3925.00, 3940.00, 3955.00, 3923.67, 3943.67])
hd_boundary = np.array([4010.00, 4060.00, 4145.00, 4195.00, 4091.74, 4111.74])
hg_boundary = np.array([4230.00, 4280.00, 4400.00, 4450.00, 4330.47, 4350.47])
hb_boundary = np.array([4750.00, 4800.00, 4920.00, 4970.00, 4851.33, 4871.33])

# Wavelenghts (in AA) of spectral lines
line_centers = np.array([3933.66, 4101.75, 4340.47, 4861.34])


# Starting constants
NUMBER_OF_ITERATIONS = 1000
#NUMBER_OF_THREADS = cpu_count()


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
        of the spectral line.
    """

    if ( (flux.size == wave.size == eflux.size) & (wave.size > 2 ) ):

        x_fit = np.linspace(min(wave), max(wave), 1000)

        f_spline = InterpolatedUnivariateSpline(wave, flux, w=eflux**-2, k=2)
        wave_offset = x_fit[np.argmin(f_spline(x_fit))]

        absolute_diff_in_line = abs(line_centers - wave_offset)
        slicer_shift = np.where(absolute_diff_in_line == min(absolute_diff_in_line))

        wave_shift = line_centers[slicer_shift] - wave_offset

        return wave_shift[0]

    else:

        return 0.0

# ------------------------------------------------------------------------------

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

    if ( (flux.size == wave.size == eflux.size) &
         (flux.size > 0 ) &
         (mask_select_conti[0].size > 0)):

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

        return flux_flip, conti_err


    else:
        return np.array([]), np.array([])

# ------------------------------------------------------------------------------

def calculate_snr(wave, flux_flip, slicing_wave):
    """
    Flux normalization using regions from Crestani+2021

    Parameters
    ----------
    wave : 'array_like, shape (N, )'
        Wavelenght in angstroms
    flux : 'array_like, shape (N, )'
        Flux in any given unit
    slicing_wave : 'array_like, shape (N, )'
        Boundaries for continuum determination

    Returns:
    -------
    snr : 'float'
        Signal-noise-ratio for given
        spectral line.
    """

    if (flux_flip.size == wave.size) & (flux_flip.size > 0 ):
        mask_select_conti = np.where(
            ((wave > slicing_wave[0]) & (wave < slicing_wave[1]))
            | ((wave > slicing_wave[2]) & (wave < slicing_wave[3])))

        snr = 1.0/np.nanstd(flux_flip[mask_select_conti])

        return snr

    else:
        return None

# ------------------------------------------------------------------------------

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

    mask_line = np.where( (wave > slicing[4]) & (wave < slicing[5]) )[0]

    if ( (flux_flip.size == wave.size == conti_err.size)
            & (flux_flip.size > 0)
            & (mask_line.size > 3) ):

        f = InterpolatedUnivariateSpline(wave, flux_flip, w=conti_err**-2, k=1)
        pseudo_ew = f.integral(slicing[4], slicing[5])

        return pseudo_ew

    else:
        return None

# ------------------------------------------------------------------------------

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

    equivalent_width, snr = ma.empty((2, number_of_lines))


    for k, (boun, reg, shift) in enumerate(zip(boundary, select_region, wave_shift)):

        if shift and (flux_r.size == wave.size == eflux.size):
            wavelenght_shifted = wave[reg] + shift

            flux_norm, eflux_norm = normalize_flux(wavelenght_shifted,
                                                   flux_r[reg],
                                                   eflux[reg],
                                                   boun)

            snr[k] = calculate_snr(wavelenght_shifted, flux_norm, boun)

            equivalent_width[k] = calculate_rr_ew(wavelenght_shifted,
                                                  flux_norm,
                                                  eflux_norm,
                                                  boun)
        else:
            snr[k], equivalent_width[k] = np.nan, np.nan

    if np.isnan(equivalent_width).all() or np.isnan(snr).all():
        return ma.ones((2, number_of_lines))*-999.

    return equivalent_width, snr

# ------------------------------------------------------------------------------

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

    if ( (flux.size == wave.size == eflux.size)
            & (flux.size > 0) ):

        flux_r = np.array([np.random.normal(flux, eflux)
                          for i in range(NUMBER_OF_ITERATIONS)])

        res = [multi_proc_function(wave, abs(flux_r[i]), eflux, boundary, select_region, wave_shift) for i in range(NUMBER_OF_ITERATIONS)]

        """ 
        with parallel_backend('multiprocessing'):
            res = Parallel(n_jobs=NUMBER_OF_THREADS)((delayed(multi_proc_function))
                (wave, abs(flux_r[i]), eflux, boundary, select_region, wave_shift)
                for i in range(NUMBER_OF_ITERATIONS))

        """

        res = np.array(res)

        ew_for_all_lines = res[:,0]
        snr_for_all_lines = res[:,1]


        return ew_for_all_lines, snr_for_all_lines

    else:
        return np.ones((1000,4))*9999, np.ones((1000,4))*-1


# ----------------------------------------------------------------------------

def calculation_of_metallicity(ew_ls):
    """
    Calculating metallicity based on relations from Crestani+2021.

    Parameters
    ----------
    ew_ls : 'array_like, shape (NUMBER_OF_ITERATIONS, 4)'
        Pseudo-equivalent widths of all lines.

    Returns:
    -------
    feh : 'float'
        Average metallicity based on
        relations Crestani+2021.
    efeh : 'float'
        Standard deviation of metallicity
        based on relations Crestani+2021.
    """

    if ew_ls.ndim < 2:
        return (-999., -999.)


    if ( (np.nansum(ew_ls[:,0]) > 0.) and
           (np.nansum(ew_ls[:,1]) > 0.) and
           (np.nansum(ew_ls[:,2]) > 0.) and
           (np.nansum(ew_ls[:,3]) > 0.) ):

           feh = C0 + C1*ew_ls[:,0] + C2*ew_ls[:,1] + C3*ew_ls[:,2] + C4*ew_ls[:,3]


    elif ( (np.nansum(ew_ls[:,0]) > 0.) and
           (np.nansum(ew_ls[:,1]) > 0.) and
           (np.nansum(ew_ls[:,2]) > 0.) and
           (np.nansum(ew_ls[:,3]) == 0.) ):

           feh = C0a + C1a*ew_ls[:,0] + C2a*ew_ls[:,1] + C3a*ew_ls[:,2]


    elif ( (np.nansum(ew_ls[:,0]) > 0.) and
           (np.nansum(ew_ls[:,1]) > 0.) and
           (np.nansum(ew_ls[:,2]) == 0.) and
           (np.nansum(ew_ls[:,3]) > 0.) ):

           feh = C0b + C1b*ew_ls[:,0] + C2b*ew_ls[:,1] + C4b*ew_ls[:,3]


    elif ( (np.nansum(ew_ls[:,0]) > 0.) and
           (np.nansum(ew_ls[:,1]) == 0.) and
           (np.nansum(ew_ls[:,2]) > 0.) and
           (np.nansum(ew_ls[:,3]) > 0.) ):

           feh = C0c + C1c*ew_ls[:,0] + C3c*ew_ls[:,2] + C4c*ew_ls[:,3]


    elif ( (np.nansum(ew_ls[:,0]) > 0.) and
           (np.nansum(ew_ls[:,1]) > 0.) and
           (np.nansum(ew_ls[:,2]) == 0.) and
           (np.nansum(ew_ls[:,3]) == 0.) ):

           feh = C0d + C1d*ew_ls[:,0] + C2d*ew_ls[:,1]


    elif ( (np.nansum(ew_ls[:,0]) > 0.) and
           (np.nansum(ew_ls[:,1]) == 0.) and
           (np.nansum(ew_ls[:,2]) > 0.) and
           (np.nansum(ew_ls[:,3]) == 0.) ):

           feh = C0e + C1e*ew_ls[:,0] + C3e*ew_ls[:,2]


    elif ( (np.nansum(ew_ls[:,0]) > 0.) and
           (np.nansum(ew_ls[:,1]) == 0.) and
           (np.nansum(ew_ls[:,2]) == 0.) and
           (np.nansum(ew_ls[:,3]) > 0.) ):

           feh = C0f + C1f*ew_ls[:,0] + C4f*ew_ls[:,3]


    else:

        return (-999., -999.)


    return (np.nanmean(feh), np.nanstd(feh))


# ----------------------------------------------------------------------------

path = '/Users/zprudil/Dropbox/+++DESI/spec/'
# where = "/Users/zprudil/Dropbox/+++DESI/results/"
c = 299792.4580

def main():

    data_to_get = ascii.read('Results_list_of_LOS_for_DESI_beta.txt')

    rv_boot = data_to_get['rv_boot']
    erv_boot_spread = data_to_get['erv_boot_spread']
    filefile = data_to_get['file']
    TARGETID = data_to_get['TARGETID']

    
    soubor_to_write = open('DeltaS-DESI.txt', 'w')
    soubor_to_write.write('# TARGETID file FEH eFEH EW_Ca eEW_Ca  EW_Hd eEW_Hd  EW_Hg eEW_Hg  EW_Hb eEW_Hb\n')

    for i in range(len(filefile)):

        if erv_boot_spread[i] > 20: continue

        marking = path + filefile[i] + '+b.txt' #39633404920073817+59335.29963897+b.txt
        data = np.genfromtxt(marking)
        
        data[:,0] = (data[:,0] / ( 1.0 +  ((5.792105*10**-2)/(238.0185 - (10000.0/data[:,0])**2)) + (1.67917*10**-3)/( 57.362 - (10000.0/data[:,0])**2)))
        data[:,0] = data[:,0] * np.sqrt((1.-(rv_boot[i])/c)/(1.+(rv_boot[i])/c))
        

        #VAC, flux, err = opening(zpracovat[i])
        data[:,2][data[:,2] == 0.0] = 0.00000001
        data[:,2] = np.sqrt(1. / data[:,2])

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

        print(np.shape(ew_ls))
        ew_ls = np.array([ (np.zeros(len(ew_ls)))
                          if np.all(np.isnan(ew_ls[:,i]))
                          else ew_ls[:,i]
                          for i in range(len(shifts)) ]).T

        signal_to_noise = np.array([ (np.zeros(len(signal_to_noise)))
                                    if np.all(np.isnan(signal_to_noise[:,i]))
                                    else signal_to_noise[:,i]
                                    for i in range(len(shifts)) ]).T

        feh_val, efeh_val = calculation_of_metallicity(ew_ls)

        
        radek = "%s %s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n" %(str(TARGETID[i]), str(filefile[i]),
            feh_val, efeh_val,
            np.nanmean(ew_ls[:,0]), np.nanstd(ew_ls[:,0]),
            np.nanmean(ew_ls[:,1]), np.nanstd(ew_ls[:,1]),
            np.nanmean(ew_ls[:,2]), np.nanstd(ew_ls[:,2]),
            np.nanmean(ew_ls[:,3]), np.nanstd(ew_ls[:,3]))
        soubor_to_write.write(radek)

    soubor_to_write.close()
    return None



if __name__ == "__main__":
    main()


