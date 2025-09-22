#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

# pylint: disable=redefined-builtin
from os import path
import pathlib
from astropy.constants import si as constants
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths # pylint: disable=no-name-in-module
from scipy.stats import norm 
from ao_tools import etc

class StructType:
    pass

class AtmosphereException(Exception):
    pass

def get_default_data_path():
    data_path = f"{pathlib.Path(__file__).parent.parent.resolve()}/data/sky"

    if not path.exists(data_path):
        raise AtmosphereException('Data files missing. See "data/README.txt" for more details.')

    return data_path

def _get_airmass_for_filename(location, airmass):
    match location:
        case 'MaunaKea':
            if airmass < 1.25:
                return 1.0
            elif airmass < 1.75:
                return 1.5
            else:
                return 2.0
        case 'Paranal':
            if airmass < 1.05:
                return 1.0
            elif airmass < 1.25:
                return 1.15
            elif airmass < 1.75:
                return 1.5
            else:
                return 2.0

    return airmass

def get_wavelength_unit():
    return u.nm

def get_emission_unit():
    return u.ph / u.s / u.m**2 / u.arcsec**2 / u.nm

def _set_wvl_unit(wvl, wvl_range, return_unitless=False):
    wvl_unit = get_wavelength_unit()

    if wvl is not None:
        if isinstance(wvl, u.Quantity):
            if return_unitless:
                wvl = wvl.to_value(wvl_unit)
            else:
                wvl = wvl.to(wvl_unit)
        else:
            if not return_unitless:
                wvl = wvl * wvl_unit

    if wvl_range is not None:
        if isinstance(wvl_range, list) and isinstance(wvl_range[0], list) and len(wvl_range[0]) == 3:
            wvl_range = u.Quantity([u.Quantity(r[1:3]) for r in wvl_range])
        elif isinstance(wvl_range, u.Quantity):
            if return_unitless:
                wvl_range = wvl_range.to_value(wvl_unit)
            else:
                wvl_range = wvl_range.to(wvl_unit)
        else:
            if isinstance(wvl_range[0], u.Quantity) or isinstance(wvl_range[0][0], u.Quantity):
                if return_unitless:
                    wvl_range = u.Quantity(wvl_range).to_value(wvl_unit)
                else:
                    wvl_range = u.Quantity(wvl_range).to(wvl_unit)
            else:
                if not return_unitless:
                    wvl_range = np.asarray(wvl_range) * wvl_unit

    return wvl, wvl_range

def load_transmission_data(wvl, instrument='GIRMOS', R=3000, zenith_angle=20*u.deg, water_vapor=1.6*u.mm, return_interp=False):
    options = etc.get_minimal_sky_options(wvl, instrument=instrument, R=R, zenith_angle=zenith_angle, water_vapor=water_vapor)
    if return_interp:
        data, interp_func = etc.load_sky_transmission(options, return_interp=True)
        return data, interp_func
    else:
        data = etc.load_sky_transmission(options)
        return data

def load_background_data(wvl, instrument='GIRMOS', R=3000, zenith_angle=20*u.deg, water_vapor=1.6*u.mm, return_interp=False):
    options = etc.get_minimal_sky_options(wvl, instrument=instrument, R=R, zenith_angle=zenith_angle, water_vapor=water_vapor)
    if return_interp:
        data, interp_func = etc.load_sky_background(options, return_interp=True)
        return data, interp_func
    else:
        data = etc.load_sky_background(options)
        return data

def get_vacuum_to_air_wavelength(wavelength):
    if isinstance(wavelength, u.Quantity):
        w = wavelength.to(u.angstrom).value
        return_as_quantity = True
    else:
        w = wavelength
        return_as_quantity = False

    # See: https://classic.sdss.org/dr7/products/spectra/vacwavelength.php
    wavelength_atm = w / (1 + 2.735182e-4 + 1.314182e2 * np.power(w,-2) + 2.76249e8 * np.power(w,-4))

    if return_as_quantity:
        return (wavelength_atm * u.angstrom).to(wavelength.unit)
    else:
        return wavelength_atm

def get_emission_line_rest_wavelengths(skip_close_doublets=False):
    lines = { # Angstrom
        'OIIa' : 3727.092,
        'OIIb' : 3729.875,
        'Hb'   : 4862.680,
        'OIIIa': 4960.295,
        'OIIIb': 5008.240,
        'NIIa' : 6549.86,
        'Ha'   : 6564.610,
        'NIIb' : 6585.27,
        'SIIa' : 6718.29,
        'SIIb' : 6732.67,
    }

    if skip_close_doublets:
        wavelength_OIIa = lines.pop('OIIa')
        wavelength_OIIb = lines.pop('OIIb')
        lines['OII'] = np.mean([wavelength_OIIa, wavelength_OIIb])

    return lines

def get_mean_transmission(transmission_data, wvl0, fwhm, truncate_sigma=4.0, is_low=False):
    orig_shape = np.shape(wvl0)
    is_scalar = (orig_shape == ())

    wvl0, _ = _set_wvl_unit(wvl0, None, return_unitless=True)                   # () or (K,)
    fwhm, _ = _set_wvl_unit(fwhm, None, return_unitless=True)                   # () or (K,)
    wvl0, fwhm = np.broadcast_arrays(np.atleast_1d(wvl0), np.atleast_1d(fwhm))  # (K,), (K,)
    sigma = fwhm * etc.Constants.fwhm2sigma                                     # (K,)

    wvl = transmission_data['wavelength']                                       # (M,)
    trans = transmission_data['transmission_lo']                                # (M,)

    # Scaled difference matrix
    delta_wvl = (wvl[None, :] - wvl0[:, None]) / sigma[:, None]                 # (K, M)

    # Truncation mask
    if truncate_sigma is not None and np.isfinite(truncate_sigma):
        mask = np.abs(delta_wvl) <= truncate_sigma
    else:
        mask = np.ones_like(delta_wvl, dtype=bool)

    # Gaussian weights
    weights = np.exp(-0.5 * delta_wvl**2)
    weights = np.where(mask, weights, 0.0)

    # Weighted mean
    num = (weights * trans[None, :]).sum(axis=1)                                # (K,)
    den = weights.sum(axis=1)                                                   # (K,)
    mean_trans = np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)

    # Shape-preserving return
    if is_scalar:
        return mean_trans.item()
    else:
        return mean_trans.reshape(orig_shape)

def get_background(interp_func, wvl, wvl_range=None):
    wvl, wvl_range = _set_wvl_unit(wvl, wvl_range)

    if wvl_range is not None:
        backgrounds = np.zeros_like(wvl.value)
        wvl_filter = (wvl >= wvl_range[0]) & (wvl <= wvl_range[1])
        backgrounds[wvl_filter] = interp_func(wvl[wvl_filter].to(u.nm).value)
    else:
        backgrounds = interp_func(wvl.to(u.nm).value)

    return backgrounds * get_emission_unit()

def find_sky_lines(
        wvl,
        emission, 
        min_photon_rate = 10 * u.photon / u.s / u.m**2 / u.arcsec**2 / u.nm
):
    # MIN PHOTON RATE:
    #
    # G * eta_sys * dLambda = 5.6e-3 (YJ) 7.3e-3 (JH) 9.2e-3 (HK) arcsec^2 m^2 nm
    # Thermal floor for GIRMOS is 0.05 e-/s
    # Flux = Thermal floor / (G * eta_sys * dLambda) = 8.9 (YJ) 6.8 (JH) 5.4 (HK) ph/s/m^2/arcsec^2/nm
    #
    peaks, _ = find_peaks(emission, height=min_photon_rate.to_value(emission.unit))
    widths, width_heights, left_ips, right_ips = peak_widths(emission, peaks, rel_height=0.5) # FWHM

    wvl_start = wvl[0]
    wvl_step = np.round(wvl[1] - wvl[0], 3)
    wvl_unit = get_wavelength_unit()
    emission_unit = get_emission_unit()

    sky_lines = Table([
            wvl[peaks],
            emission[peaks],
            widths * wvl_step,
            width_heights,
            wvl_start + left_ips * wvl_step,
            wvl_start + right_ips * wvl_step
        ], names=[
            'wavelength',
            'emission',
            'width',
            'width_height',
            'wavelength_low',
            'wavelength_high'
        ], units=[
            wvl_unit,
            emission_unit,
            wvl_unit,
            emission_unit,
            wvl_unit,
            wvl_unit
        ]
    )

    return sky_lines

def _make_2d_like(arr, target_shape_nm):
    """Broadcast arr (scalar, (N,), (M,), or (N,M)) to (N,M)."""
    N, M = target_shape_nm
    if isinstance(arr, u.Quantity):
        units = arr.unit
    else:
        units = None

    a = np.asarray(arr) if arr is not None else None
    if a is None:
        out = None
    elif a.ndim == 0:
        out = np.full((N, M), a)
    elif a.ndim == 1:
        if a.shape[0] == N:
            out = np.repeat(a[:, None], M, axis=1)
        elif a.shape[0] == M:
            out = np.repeat(a[None, :], N, axis=0)
        else:
            raise ValueError(f"1D array length {a.shape[0]} cannot broadcast to (N,M)=({N},{M}).")
    elif a.ndim == 2:
        if a.shape != (N, M):
            raise ValueError(f"2D array shape {a.shape} must equal (N,M)=({N},{M}).")
        out = a
    else:
        raise ValueError("Input must be scalar, 1D, or 2D.")

    if out is not None and units is not None:
        return out * units
    else:
        return out

def reject_emission_line(
        etc_params, flux, wvl, dispersion, eta_spat=0.01,
        snr_threshold = 3.0,
        return_snr=False,
        return_ENBW=False,
        return_signals=False,
        spec_wvl=None,
        include_fit_error_adjustment=False,
):
    """
    Reject a line if its matched-filter S/N (with source+background shot noise) < snr_threshold,
    or if it falls outside allowed ranges.

    Parameters:
      - etc_params    : parameters for ETC calculations
      - flux          : total line flux [erg/s/cm^2] (scalar, (N,), (M,), or (N,M))
      - wvl           : wavelength(s) to check (scalar, (N,), or (N,M))
      - dispersion    : intrinsic line dispersion (scalar, (N,), (M,), or (N,M))
      - eta_spat      : spatial efficiency (fraction of light in aperture)
      - snr_threshold : minimum S/N to accept, default=3 is minimum required to detect line flux
      - return_snr
      - return_ENBW
      - return_signals
      - spec_wvl      : used only if return_rates=True; wavelength grid of the full spectrum for computing Ndot_sky and Ndot_source [nm]
      - include_fit_error_adjustment: if True, include the fit error adjustment factor in the noise calculation
    """

    wvl, wvl_ranges = _set_wvl_unit(wvl, etc_params['bands'])
    dispersion      = dispersion.to(u.km/u.s)

    # Standardized Gaussian sampling grid
    # u_std, w_std = etc.precompute_gaussian_standardized_grid()

    # Coerce wvl to (N,M)
    orig_shape = wvl.shape
    if wvl.ndim == 0:
        wvl = wvl.reshape(1, 1)
    elif wvl.ndim == 1:
        if np.size(snr_threshold) == 1:
            wvl = wvl[:, None]
        else:
            wvl = wvl[None, :]
    elif wvl.ndim > 2:
        raise ValueError("wvl must be scalar, 1D, or 2D.")
    N, M = wvl.shape

    # Broadcast fwhm and A (total observed line rate per spaxel) to (N,M)
    dispersion     = _make_2d_like(dispersion, (N, M))
    flux           = _make_2d_like(flux, (N, M))
    eta_spat       = _make_2d_like(eta_spat, (N, M))
    snr_thresholds = np.broadcast_to(snr_threshold, (M,))

    # Create results mask
    rejects = np.ones((N, M), dtype=bool)
    snr = np.full((N, M), np.nan)
    
    if return_signals:
        ENBW_wvl = np.full((N, M), np.nan) * get_wavelength_unit()

    if return_signals:
        if spec_wvl is None:
            raise ValueError("Either spec_wvl or sky_data must be provided if return_counts=True.")
        N_signal = np.full((len(spec_wvl), M), np.nan) * u.electron
        N_noise  = np.full((len(spec_wvl), M), np.nan) * u.electron

    for j in range(M):
        if wvl_ranges is None:
            valid = np.ones(wvl.shape[0], dtype=bool)
        elif wvl_ranges.ndim == 1:
            lo, hi = wvl_ranges[0], wvl_ranges[1]
            valid = (wvl[:, j] >= lo) & (wvl[:, j] <= hi)
        elif wvl_ranges.ndim == 2 and wvl_ranges.shape[1] == 2:
            x  = wvl[:, j][..., None]      # (N, 1)
            lo = wvl_ranges[:, 0]          # (K,)
            hi = wvl_ranges[:, 1]          # (K,)
            valid = ((x >= lo) & (x <= hi)).any(axis=-1)  # (N,)
        else:
            raise ValueError("ranges must be None, [lo, hi], or (K, 2).")

        data = {
            'flux': flux[:,j],
            'wvl0': wvl[:,j],
            'dispersion': dispersion[:,j],
            'eta_spat': eta_spat[:,j],
        }

        snr[:, j], etc_details = etc.estimate_emission_line_flux_fit_snr(etc_params, data, return_etc_details=True)
        rejects[:, j] = (snr[:, j] < snr_thresholds[j]) | (~valid)

        if return_ENBW:
            ENBW_wvl[:, j] = etc_details['ENBW'] * etc_details['wvl_step']  # [nm]

        if return_signals:
            spec_wvl, _ = _set_wvl_unit(spec_wvl, None)
            N_signal[:, j], N_noise[:, j] = etc.estimate_signal_spectrum(spec_wvl, etc_details, include_fit_error_adjustment=include_fit_error_adjustment)

    # Match caller's input shape
    return_values = []
    if len(orig_shape) == 0:      # scalar
        return_values.append(rejects.item())
    elif len(orig_shape) == 1:    # (N,)
        return_values.append(rejects.flatten())
    else:
        return_values.append(rejects)

    # Return S/N if requested
    if return_snr:
        if len(orig_shape) == 0:      # scalar
            return_values.append(snr.item())
        elif len(orig_shape) == 1:    # (N,)
            return_values.append(snr.flatten())
        else:
            return_values.append(snr)

    # Return sigma_eff if requested
    if return_ENBW:
        if len(orig_shape) == 0:      # scalar
            return_values.append(ENBW_wvl.item())
        elif len(orig_shape) == 1:    # (N,)
            return_values.append(ENBW_wvl.flatten())
        else:
            return_values.append(ENBW_wvl)

    # Compute counts if requested
    if return_signals:
        return_values.append(N_signal)
        return_values.append(N_noise)

    if len(return_values) == 1:
        return return_values[0]
    else:
        return tuple(return_values)

#region Deprecate

def load_transmission_data_hi_old(location, airmass, data_path = None):
    if data_path is None:
        data_path = get_default_data_path()

    match location:
        case 'MaunaKea':
            # From: https://www.gemini.edu/observing/telescopes-and-sites/sites#Transmission
            #
            # The infrared spectra of the atmospheric transmission above Mauna Kea that are used
            # in the Integration Time Calculators have been generated using the ATRAN modelling
            # software (Lord, S.D. 1992, NASA Technical Memor. 103957) and are presented separately
            # for the near-IR and mid-IR. Ascii data files of these spectra are available below.
            #
            # Column 1: wavelength [micron]
            # Column 2: transmission [%]
            transmission_data = ascii.read(f"{data_path}/mk_trans_zm_10_{_get_airmass_for_filename(location, airmass)*10:.0f}.dat", names=['wavelength', 'transmission'])
            transmission_data['wavelength'] *= 1e3 # micron -> nm
        case 'Paranal':
            # From: https://www.eso.org/sci/facilities/eelt/science/drm/tech_data/background/
            #
            # Column 1: wavelength [um]
            # Column 2: transmission [%]
            transmission_data = ascii.read(f"{data_path}/paranal_trans_airm{_get_airmass_for_filename(location, airmass):.2f}_wav00.4-03.0.dat", names=['wavelength', 'transmission'])
            transmission_data['wavelength'] *= 1e3 # um -> nm
        case _:
            raise AtmosphereException('Unknown location')

    transmission_data['wavelength'].unit = get_wavelength_unit()

    return transmission_data

def load_background_data_hi_old(location, airmass, data_path = None):
    if data_path is None:
        data_path = get_default_data_path()

    match location:
        case 'MaunaKea' | 'Paranal':
            # From: https://www.gemini.edu/observing/telescopes-and-sites/sites#IRSky
            #
            # The files were manufactured starting from the sky transmission files generated
            # by ATRAN (Lord, S. D., 1992, NASA Technical Memorandum 103957). These files were
            # subtracted from unity to give an emissivity and then multiplied by a blackbody
            # function of temperature 273 for Mauna Kea and 280 for Cerro Pachon. To these were
            # added the OH emission spectrum (available from the European Southern Observatory's
            # ISAAC web pages) a set of O2 lines near 1.3 microns with estimated strengths based
            # on observations at Mauna Kea, and the dark sky continuum (in part zodiacal light),
            # approximated as a 5800K gray body times the atmospheric transmission and scaled to
            # produce 18.2 mag/arcsec^2 in the H band, as observed on Mauna Kea by
            # Maihara et al. (1993 PASP, 105, 940).
            #
            # Any use of the data in these tables should reference Lord (1992) and acknowledge Gemini Observatory.
            #
            # Column 1: wavelength [nm]
            # Column 2: emission [ph/sec/m^2/arcsec^2/nm]
            background_data = ascii.read(f"{data_path}/mk_skybg_zm_10_{_get_airmass_for_filename(location, airmass)*10:.0f}_ph.dat", names=['wavelength', 'emission'])
        # case 'Paranal':
        #     # From: https://www.eso.org/sci/facilities/eelt/science/drm/tech_data/background/

        #     # Column 1: wavelength [um]
        #     # Column 2: emission [photons/s/m^2/arcsec^2]
        #     sky_background = ascii.read(f"{data_path}/paranal_optical_ir_sky_lines.dat", names=['wavelength', 'emission'])
        #     sky_background['wavelength'] *= 1e3  # um -> nm
        #     sky_background['emission']   /= (sky_background['wavelength']/10/spectral_resolving_power) # [photons/s/m^2/arcsec^2] -> [photons/s/m^2/arcsec^2/nm]
        case _:
            raise AtmosphereException('Unknown location')

    background_data['wavelength'].unit = get_wavelength_unit()
    background_data['emission'].unit = get_emission_unit()

    return background_data

def _create_interp_func(data, field_name):
    return interp1d(data['wavelength'], data[field_name], kind='linear', bounds_error=False, fill_value=np.nan)

def _convolve_with_gaussian(R, data_hi, field_name, field_unit=None, wvl_range=None, wvl_mean=None, return_interp=False):
    _, wvl_range = _set_wvl_unit(None, wvl_range, return_unitless=True)

    wvl = data_hi['wavelength']
    value_hi = data_hi[field_name]

    wvl_step = np.round(wvl[1] - wvl[0], 3)
    if wvl_mean is None:
        wvl_mean = np.mean(wvl_range if wvl_range is not None else wvl)
    else:
        wvl_mean = wvl_mean.to_value(wvl.unit)
    dlambda  = wvl_mean / R

    if wvl_range is not None and not return_interp:
        wvl_filter_wide = (wvl >= wvl_range[0] - dlambda*10) & (wvl <= wvl_range[1] + dlambda*10)

        if np.sum(wvl_filter_wide) == 0:
            return None

        wvl = wvl[wvl_filter_wide]
        value_hi = value_hi[wvl_filter_wide]

    sigma = dlambda * etc.Constants.fwhm2sigma
    fudge_factor = 1.2 # Compensates for underestimation of total sky background power due to low resolution of input data
    sigma_bins = sigma / wvl_step / fudge_factor
    value_lo = gaussian_filter1d(value_hi, sigma_bins, mode='reflect')
    if field_unit is not None:
        value_lo *= field_unit

    if wvl_range is not None:
        wvl_filter_narrow = (wvl >= wvl_range[0]) & (wvl <= wvl_range[1])
        wvl = wvl[wvl_filter_narrow]
        value_lo = value_lo[wvl_filter_narrow]

    data_lo = Table([wvl, value_lo], names=['wavelength', field_name], units=[get_wavelength_unit(), field_unit])

    if return_interp:
        return data_lo, _create_interp_func(data_lo, field_name)
    else:
        return data_lo

def get_background_data_lo_old(background_data, R, wvl_range=None, wvl_mean=None, return_interp=False):
    return _convolve_with_gaussian(R, background_data, "emission", field_unit=get_emission_unit(), wvl_range=wvl_range, wvl_mean=wvl_mean, return_interp=return_interp)

def find_sky_lines_old(
        background_data, 
        min_photon_rate=10.0 # ph/sec/m^2/arcsec^2/nm
):
    # MIN PHOTON RATE:
    #
    # G * eta_sys * dLambda = 5.6e-3 (YJ) 7.3e-3 (JH) 9.2e-3 (HK) arcsec^2 m^2 nm
    # Thermal floor for GIRMOS is 0.05 e-/s
    # Flux = Rate / (G * eta_sys * dLambda) = 8.9 (YJ) 6.8 (JH) 5.4 (HK) ph/s/m^2/arcsec^2/nm
    #

    peaks, _ = find_peaks(background_data['emission'], height=min_photon_rate)
    widths, width_heights, left_ips, right_ips = peak_widths(background_data['emission'], peaks, rel_height=0.5) # FWHM

    wvl_start = background_data['wavelength'][0]
    wvl_step = np.round(background_data['wavelength'][1] - background_data['wavelength'][0], 3)
    wvl_unit = get_wavelength_unit()
    emission_unit = get_emission_unit()

    sky_lines = Table([
            background_data['wavelength'][peaks],
            background_data['emission'][peaks],
            widths * wvl_step,
            width_heights,
            wvl_start + left_ips * wvl_step,
            wvl_start + right_ips * wvl_step
        ], names=[
            'wavelength',
            'emission',
            'width',
            'width_height',
            'wavelength_low',
            'wavelength_high'
        ], units=[
            wvl_unit,
            emission_unit,
            wvl_unit,
            emission_unit,
            wvl_unit,
            wvl_unit
        ]
    )

    return sky_lines

def reject_emission_line_old(
        background_data,
        transmission_data,
        wvl,
        fwhm,
        R,
        allowed_wavelength_range = None,
        trans_minimum = 1.0,
        avoid_dLambda_multiple = 1.0,
        min_photon_rate = 10.0 # ph/sec/m^2/arcsec^2/nm
):
    if hasattr(wvl, 'shape'):
        N = wvl.size
        rejects = np.ones(wvl.shape, dtype=np.bool)
    else:
        N = len(wvl)
        rejects = np.ones((N), dtype=np.bool)

    for i in np.arange(N):
        if N == 1:
            current_wavelength = wvl
        else:
            current_wavelength = wvl[i]

        if np.size(fwhm) == 1:
            current_fwhm = fwhm
        else:
            current_fwhm = fwhm[i]

        reject = False

        if current_wavelength == 0:
            reject = True
        else:
            dwavelength = np.sqrt(np.power(current_wavelength/R,2) + np.power(current_fwhm,2))
            wavelength_range = current_wavelength + 10*dwavelength*np.array([-0.5, 0.5])
            trans = get_mean_transmission(transmission_data, current_wavelength, current_fwhm, R=R)

            if trans < trans_minimum:
                reject = True

            if not reject and allowed_wavelength_range is not None:
                reject = True
                if np.ndim(allowed_wavelength_range) == 1:
                    if ((current_wavelength - dwavelength) >= allowed_wavelength_range[0] and (current_wavelength + dwavelength) <= allowed_wavelength_range[1]):
                        reject = False
                else:
                    num_ranges = np.shape(allowed_wavelength_range)[0]
                    for j in np.arange(num_ranges):
                        if ((current_wavelength - dwavelength) >= allowed_wavelength_range[j,0] and (current_wavelength + dwavelength) <= allowed_wavelength_range[j,1]):
                            reject = False
                            break

            if not reject:
                wvl_start = background_data['wavelength'].quantity[0]
                wvl_end   = background_data['wavelength'].quantity[-1]
                if wavelength_range[0] < wvl_start or wavelength_range[1] > wvl_end:
                    reject = True

            if not reject:
                background_low_res = get_background_data_lo(background_data, R, wavelength_range)
                sky_lines = find_sky_lines(background_low_res, min_photon_rate)

                line_wavelength_low  = current_wavelength - dwavelength * avoid_dLambda_multiple
                line_wavelength_high = current_wavelength + dwavelength * avoid_dLambda_multiple
                reject = np.any((sky_lines['wavelength_low' ] <= line_wavelength_high) & (sky_lines['wavelength_high'] >= line_wavelength_low))

        if N == 1:
            rejects = reject
        else:
            rejects[i] = reject

    return rejects

#endregion
